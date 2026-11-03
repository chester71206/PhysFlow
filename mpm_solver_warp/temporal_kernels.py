"""Out-of-place versions of the existing MPM stages (same physics).

The reference solver and verified search deliberately do not import this module.
Only elastic, plasticine without damage/hardening, and existing Newtonian modes
are supported. Every array written by a taped launch has fresh storage.
"""
import warp as wp
from mpm_solver_warp.mpm_solver_warp import (
    MPMModelStruct, MPMStateStruct, Dirichlet_collider,
    SCALE_YS, compute_dweight, kirchoff_stress_FCR, kirchoff_stress_StVK,
    kirchoff_stress_herschel, kirchoff_stress_newtonian_rate,
)


@wp.struct
class StressOutput:
    particle_F: wp.array(dtype=wp.mat33)
    particle_stress: wp.array(dtype=wp.mat33)


@wp.struct
class ParticleOutput:
    particle_x: wp.array(dtype=wp.vec3)
    particle_v: wp.array(dtype=wp.vec3)
    particle_C: wp.array(dtype=wp.mat33)
    particle_F_trial: wp.array(dtype=wp.mat33)
    particle_F_N_trial: wp.array(dtype=wp.mat33)
    particle_cov: wp.array(dtype=float)


@wp.struct
class TransferModel:
    inv_dx: float
    dx: float
    grid_dim_x: int
    grid_dim_y: int
    grid_dim_z: int
    update_cov_with_F: int


@wp.func
def transfer_dweight(model: TransferModel, w: wp.mat33, dw: wp.mat33, i: int, j: int, k: int):
    return wp.vec3(dw[0, i] * w[1, j] * w[2, k],
                   w[0, i] * dw[1, j] * w[2, k],
                   w[0, i] * w[1, j] * dw[2, k]) * model.inv_dx


@wp.kernel
def stress_out(state: MPMStateStruct, out: StressOutput, model: MPMModelStruct):
    p = wp.tid()
    F = state.particle_F[p]
    stress = state.particle_stress[p]
    if state.particle_selection[p] == 0:
        F = state.particle_F_trial[p]
        if model.material == 4:
            F = plasticine_return_mapping_readonly(F, model, p)
        J = wp.determinant(F)
        U = wp.mat33(0.0)
        V = wp.mat33(0.0)
        sig = wp.vec3(0.0)
        wp.svd3(F, U, sig, V)
        if model.material == 0:
            stress = kirchoff_stress_FCR(F, U, V, J, model.mu[p], model.lam[p])
        elif model.material == 4:
            stress = kirchoff_stress_StVK(F, U, V, sig, model.mu[p], model.lam[p])
        elif model.material == 5:
            if model.newtonian_rate_model == 1:
                stress = kirchoff_stress_newtonian_rate(J, state.particle_C[p], model.mu[p], model.lam[p])
            else:
                stress = kirchoff_stress_herschel(F, J, model.mu[p], model.lam[p])
        stress = (stress + wp.transpose(stress)) / 2.0
    out.particle_F[p] = F
    out.particle_stress[p] = stress


@wp.kernel
def grid_scale_out(before: wp.array(dtype=wp.vec3, ndim=3),
                   after: wp.array(dtype=wp.vec3, ndim=3), scale: float):
    i, j, k = wp.tid()
    after[i, j, k] = before[i, j, k] * scale


@wp.kernel
def boundary_out(before: wp.array(dtype=wp.vec3, ndim=3),
                 after: wp.array(dtype=wp.vec3, ndim=3), model: MPMModelStruct,
                 param: Dirichlet_collider, kind: int, time: float):
    i, j, k = wp.tid()
    v = before[i, j, k]
    if time >= param.start_time and time < param.end_time:
        if kind == 0:  # Existing bounding-box conditions, same order.
            if i < 3 and v[0] < 0.0:
                v = wp.vec3(0.0, v[1], v[2])
            if i >= model.grid_dim_x - 3 and v[0] > 0.0:
                v = wp.vec3(0.0, v[1], v[2])
            if float(j) < param.threshold and v[1] < 0.0:
                v = wp.vec3(v[0], 0.0, v[2])
            if j >= model.grid_dim_y - 3 and v[1] > 0.0:
                v = wp.vec3(v[0], 0.0, v[2])
            if k < 3 and v[2] < 0.0:
                v = wp.vec3(v[0], v[1], 0.0)
            if k >= model.grid_dim_z - 3 and v[2] > 0.0:
                v = wp.vec3(v[0], v[1], 0.0)
        else:  # Preserve even the reference's zero-velocity slip branch.
            offset = wp.vec3(float(i) * model.dx - param.point[0],
                             float(j) * model.dx - param.point[1],
                             float(k) * model.dx - param.point[2])
            if wp.dot(offset, param.normal) < 0.0:
                if param.surface_type == 11:
                    if float(k) * model.dx < 0.4 or float(k) * model.dx > 0.53:
                        v = wp.vec3(0.0)
                    else:
                        v = wp.vec3(v[0], 0.0, v[2]) * 0.3
                else:
                    v = wp.vec3(0.0)
    after[i, j, k] = v

@wp.func
def plasticine_return_mapping_readonly(
    F_trial: wp.mat33, model: MPMModelStruct, p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[p] * epsilon + model.lam[p] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    
    if model.yield_stress[0] <= 0:
        return F_trial
    # Elastic trial states inside the yield surface must be preserved. Without
    # this guard delta_gamma is negative, so the projection amplifies small
    # strains (including SVD roundoff at identity) toward the yield surface.
    if wp.length(cond) <= SCALE_YS * model.yield_stress[0]:
        return F_trial
    epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
    epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
    delta_gamma = epsilon_hat_norm - SCALE_YS * model.yield_stress[0] / (2.0 * model.mu[p])
    epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
    sig_elastic = wp.mat33(
        wp.exp(epsilon[0]),
        0.0,
        0.0,
        0.0,
        wp.exp(epsilon[1]),
        0.0,
        0.0,
        0.0,
        wp.exp(epsilon[2]),
    )
    F_elastic = U * sig_elastic * wp.transpose(V)
    return F_elastic
    


@wp.func
def update_cov_out(state: MPMStateStruct, out: ParticleOutput, p: int, grad_v: wp.mat33, dt: float):
    cov_n = wp.mat33(0.0)
    cov_n[0, 0] = state.particle_cov[p * 6]
    cov_n[0, 1] = state.particle_cov[p * 6 + 1]
    cov_n[0, 2] = state.particle_cov[p * 6 + 2]
    cov_n[1, 0] = state.particle_cov[p * 6 + 1]
    cov_n[1, 1] = state.particle_cov[p * 6 + 3]
    cov_n[1, 2] = state.particle_cov[p * 6 + 4]
    cov_n[2, 0] = state.particle_cov[p * 6 + 2]
    cov_n[2, 1] = state.particle_cov[p * 6 + 4]
    cov_n[2, 2] = state.particle_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    out.particle_cov[p * 6] = cov_np1[0, 0]
    out.particle_cov[p * 6 + 1] = cov_np1[0, 1]
    out.particle_cov[p * 6 + 2] = cov_np1[0, 2]
    out.particle_cov[p * 6 + 3] = cov_np1[1, 1]
    out.particle_cov[p * 6 + 4] = cov_np1[1, 2]
    out.particle_cov[p * 6 + 5] = cov_np1[2, 2]



@wp.kernel
def g2p_out(state: MPMStateStruct, out: ParticleOutput, model: TransferModel, dt: float):
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
                    grid_v = state.grid_v_out[ix, iy, iz]
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
