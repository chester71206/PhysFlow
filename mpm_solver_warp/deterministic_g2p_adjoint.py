"""Experimental fixed-order G2P grid adjoint; original forward physics unchanged.

The G2P primal reads the same grid values through a per-particle buffer. The
buffer VJP is unique per particle; its scatter to shared grid cells is reduced
in fixed order. This is an isolated diagnostic, never imported by reference.
"""
import torch
import warp as wp
from mpm_solver_warp.mpm_solver_warp import MPMStateStruct, MPMModelStruct
from mpm_solver_warp.temporal_kernels import ParticleOutput, TransferModel, transfer_dweight, update_cov_out
from mpm_solver_warp.deterministic_p2g_diagnostic import contribution_keys
from mpm_solver_warp.temporal_rollout import fresh

@wp.kernel
def g2p_local_out(state: MPMStateStruct, out: ParticleOutput, model: TransferModel, dt: float, local_grid: wp.array(dtype=wp.vec3)):
    p = wp.tid()
    if state.particle_selection[p] == 0:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
        new_v = wp.vec3(0.0, 0.0, 0.0)
        new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    # avoid out of range
                    ix = wp.clamp(ix, 0, model.grid_dim_x-1)
                    iy = wp.clamp(iy, 0, model.grid_dim_y-1)
                    iz = wp.clamp(iz, 0, model.grid_dim_z-1)

                    dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    grid_v = local_grid[p * 27 + i * 9 + j * 3 + k]
                    new_v = new_v + grid_v * weight
                    new_C = new_C + wp.outer(grid_v, dpos) * (
                        weight * model.inv_dx * 4.0
                    )
                    dweight = transfer_dweight(model, w, dw, i, j, k)
                    new_F = new_F + wp.outer(grid_v, dweight)

        out.particle_v[p] = new_v
        out.particle_x[p] = state.particle_x[p] + dt * new_v
        out.particle_C[p] = new_C
        I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        F_tmp = (I33 + new_F * dt) * state.particle_F[p] #- state.particle_F_N[p])
        out.particle_F_trial[p] = F_tmp
        # add
        F_N_tmp = (I33 + new_F * dt) * state.particle_F_N[p]
        out.particle_F_N_trial[p] = F_N_tmp

        if model.update_cov_with_F:
            update_cov_out(state, out, p, new_F, dt)

    else:
        out.particle_x[p] = state.particle_x[p]
        out.particle_v[p] = state.particle_v[p]
        out.particle_C[p] = state.particle_C[p]
        out.particle_F_trial[p] = state.particle_F_trial[p]
        out.particle_F_N_trial[p] = state.particle_F_N_trial[p]
        if model.update_cov_with_F:
            for i in range(6):
                out.particle_cov[6 * p + i] = state.particle_cov[6 * p + i]


@wp.kernel
def gather_local_grid(grid: wp.array(dtype=wp.vec3, ndim=3),
                      keys: wp.array(dtype=wp.int32),
                      local: wp.array(dtype=wp.vec3), model: MPMModelStruct):
    q = wp.tid()
    key = keys[q]
    if key < model.grid_dim_x * model.grid_dim_y * model.grid_dim_z:
        ix = key // (model.grid_dim_y * model.grid_dim_z)
        rest = key - ix * model.grid_dim_y * model.grid_dim_z
        iy = rest // model.grid_dim_z
        iz = rest - iy * model.grid_dim_z
        local[q] = grid[ix, iy, iz]
    else:
        local[q] = wp.vec3(0.0)


@wp.kernel
def reduce_local_grid_adjoint(local_grad: wp.array(dtype=wp.vec3),
                              grid_grad: wp.array(dtype=wp.vec3, ndim=3),
                              keys: wp.array(dtype=wp.int32),
                              values: wp.array(dtype=wp.int32),
                              count: int, max_segment: int,
                              model: MPMModelStruct):
    first = wp.tid()
    key = keys[first]
    if key >= model.grid_dim_x * model.grid_dim_y * model.grid_dim_z:
        return
    if first > 0:
        if keys[first - 1] == key:
            return
    total = wp.vec3(0.0)
    for offset in range(max_segment):
        q = first + offset
        if q < count:
            if keys[q] == key:
                total = total + local_grad[values[q]]
    ix = key // (model.grid_dim_y * model.grid_dim_z)
    rest = key - ix * model.grid_dim_y * model.grid_dim_z
    iy = rest // model.grid_dim_z
    iz = rest - iy * model.grid_dim_z
    grid_grad[ix, iy, iz] = grid_grad[ix, iy, iz] + total


def launch(stage, particle_output, transfer_model, material_model, dt, count):
    """Launch the same G2P primal with a deterministic shared-grid VJP."""
    device = stage.particle_x.device
    n = count * 27
    keys = wp.empty(n, dtype=wp.int32, device=device)
    values = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(contribution_keys, n, inputs=[stage, material_model, keys, values],
              device=device, record_tape=False)
    local = fresh(wp.empty(n, dtype=wp.vec3, device=device))
    wp.launch(gather_local_grid, n, inputs=[stage.grid_v_out, keys, local, material_model],
              device=device, record_tape=False)
    wp.synchronize_device(device)
    key_tensor = wp.to_torch(keys)
    order = torch.argsort(key_tensor, stable=True)
    cell_count = material_model.grid_dim_x * material_model.grid_dim_y * material_model.grid_dim_z
    sorted_keys_tensor = torch.cat((key_tensor[order],
                                    torch.tensor([cell_count], device=key_tensor.device,
                                                 dtype=torch.int32)))
    sorted_values_tensor = torch.cat((order.to(torch.int32),
                                      torch.zeros(1, device=key_tensor.device,
                                                  dtype=torch.int32)))
    sorted_keys = wp.from_torch(sorted_keys_tensor.contiguous(), dtype=wp.int32)
    sorted_values = wp.from_torch(sorted_values_tensor.contiguous(), dtype=wp.int32)
    max_segment = int(torch.unique_consecutive(sorted_keys_tensor[:n],
                                                return_counts=True)[1].max())
    tape = wp.context.runtime.tape
    if tape is None:
        raise RuntimeError('Deterministic G2P adjoint requires a Warp tape')

    def backward_gather():
        wp.launch(reduce_local_grid_adjoint, n,
                  inputs=[local.grad, stage.grid_v_out.grad, sorted_keys,
                          sorted_values, n, max_segment, material_model],
                  device=device, record_tape=False)

    tape.record_func(backward_gather, [stage.grid_v_out, local])
    wp.launch(g2p_local_out, count,
              inputs=[stage, particle_output, transfer_model, dt, local], device=device)
