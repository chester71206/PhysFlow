"""Opt-in fixed-order P2G experiment; original solver never imports this file.

Sort integer (grid cell, particle/stencil index) pairs, then let exactly one
thread sum each cell's contributions in sorted order. The contribution formula
matches the existing P2G kernel; sort keys are discrete as in the original
cell assignment. This is a diagnostic implementation, not yet a promoted path.
"""
import warp as wp

from mpm_solver_warp.mpm_solver_warp import MPMModelStruct, MPMStateStruct, compute_dweight


@wp.kernel
def contribution_keys(state: MPMStateStruct, model: MPMModelStruct,
                      keys: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.int32)):
    q = wp.tid()
    p = q // 27
    local = q - p * 27
    i = local // 9
    j = (local - i * 9) // 3
    k = local - i * 9 - j * 3
    cell_count = model.grid_dim_x * model.grid_dim_y * model.grid_dim_z
    key = cell_count
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        ix = wp.clamp(wp.int(grid_pos[0] - 0.5) + i, 0, model.grid_dim_x - 1)
        iy = wp.clamp(wp.int(grid_pos[1] - 0.5) + j, 0, model.grid_dim_y - 1)
        iz = wp.clamp(wp.int(grid_pos[2] - 0.5) + k, 0, model.grid_dim_z - 1)
        key = (ix * model.grid_dim_y + iy) * model.grid_dim_z + iz
    keys[q] = key
    values[q] = q


@wp.kernel
def reduce_sorted_contributions(state: MPMStateStruct, model: MPMModelStruct,
                                dt: float, keys: wp.array(dtype=wp.int32),
                                values: wp.array(dtype=wp.int32), count: int,
                                max_segment: int):
    first = wp.tid()
    key = keys[first]
    if key >= model.grid_dim_x * model.grid_dim_y * model.grid_dim_z:
        return
    if first > 0:
        if keys[first - 1] == key:
            return
    grid_m = float(0.0)
    grid_v = wp.vec3(0.0)
    # Warp 0.10 does not support reverse-mode through most while loops.
    # A runtime-bounded for loop preserves the same fixed summation order.
    for offset in range(max_segment):
        q = first + offset
        if q < count:
            if keys[q] == key:
                entry = values[q]
                p = entry // 27
                local = entry - p * 27
                i = local // 9
                j = (local - i * 9) // 3
                k = local - i * 9 - j * 3
                grid_pos = state.particle_x[p] * model.inv_dx
                base_pos_x = wp.int(grid_pos[0] - 0.5)
                base_pos_y = wp.int(grid_pos[1] - 0.5)
                base_pos_z = wp.int(grid_pos[2] - 0.5)
                fx = grid_pos - wp.vec3(wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z))
                wa = wp.vec3(1.5) - fx
                wb = fx - wp.vec3(1.0)
                wc = fx - wp.vec3(0.5)
                w = wp.mat33(wp.cw_mul(wa, wa) * 0.5,
                             wp.vec3(0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                             wp.cw_mul(wc, wc) * 0.5)
                dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
                dpos = (wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx) * model.dx
                weight = w[0, i] * w[1, j] * w[2, k]
                dweight = compute_dweight(model, w, dw, i, j, k)
                C = state.particle_C[p]
                C = (1.0 - model.rpic_damping) * C + model.rpic_damping / 2.0 * (C - wp.transpose(C))
                if model.rpic_damping < -0.001:
                    C = wp.mat33(0.0)
                elastic_force = -state.particle_vol[p] * state.particle_stress[p] * dweight
                contribution = weight * state.particle_mass[p] * (state.particle_v[p] + C * dpos) + dt * elastic_force
                grid_v = grid_v + contribution
                grid_m = grid_m + weight * state.particle_mass[p]
    ix = key // (model.grid_dim_y * model.grid_dim_z)
    rest = key - ix * model.grid_dim_y * model.grid_dim_z
    iy = rest // model.grid_dim_z
    iz = rest - iy * model.grid_dim_z
    state.grid_v_in[ix, iy, iz] = grid_v
    state.grid_m[ix, iy, iz] = grid_m


def launch(state, model, dt, particle_count, record_tape=True):
    import torch
    count = particle_count * 27
    keys = wp.empty(count, dtype=wp.int32, device=state.particle_x.device)
    values = wp.empty(count, dtype=wp.int32, device=state.particle_x.device)
    wp.launch(contribution_keys, count, inputs=[state, model, keys, values],
              device=state.particle_x.device, record_tape=record_tape)
    # Warp and PyTorch may use different streams. The integer keys must be
    # complete before PyTorch reads them, and the sorted tensors must remain
    # live until Warp finishes the reduction launch.
    wp.synchronize_device(state.particle_x.device)
    key_tensor = wp.to_torch(keys)
    order = torch.argsort(key_tensor, stable=True)
    cell_count = model.grid_dim_x * model.grid_dim_y * model.grid_dim_z
    sorted_keys_tensor = torch.cat((key_tensor[order],
                                    torch.tensor([cell_count], device=key_tensor.device,
                                                 dtype=torch.int32)))
    sorted_values_tensor = torch.cat((order.to(torch.int32),
                                      torch.zeros(1, device=key_tensor.device,
                                                  dtype=torch.int32)))
    sorted_keys = wp.from_torch(sorted_keys_tensor.contiguous(), dtype=wp.int32)
    sorted_values = wp.from_torch(sorted_values_tensor.contiguous(), dtype=wp.int32)
    max_segment = int(torch.unique_consecutive(sorted_keys_tensor[:count],
                                                return_counts=True)[1].max())
    wp.launch(reduce_sorted_contributions, count,
              inputs=[state, model, dt, sorted_keys, sorted_values, count, max_segment],
              device=state.particle_x.device, record_tape=record_tape)
    wp.synchronize_device(state.particle_x.device)
    return sorted_keys, sorted_values
