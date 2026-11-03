"""Analytic per-particle P2G VJP for the unchanged fixed-order P2G primal.

The stencil index is discrete, as in the original kernel. Within that stencil
the position derivative uses the quadratic B-spline gradient and Hessian.
Every thread writes only its own particle gradients; grid gradients are read.
"""
import warp as wp

from mpm_solver_warp.mpm_solver_warp import MPMModelStruct, MPMStateStruct
from mpm_solver_warp.deterministic_p2g_diagnostic import launch as fixed_primal


@wp.struct
class P2GTransferModel:
    dx: float
    inv_dx: float
    grid_dim_x: int
    grid_dim_y: int
    grid_dim_z: int
    rpic_damping: float


def transfer_model(model):
    result = P2GTransferModel()
    for name in result._cls.vars:
        setattr(result, name, getattr(model, name))
    return result


@wp.kernel(enable_backward=False)
def p2g_particle_vjp(state: MPMStateStruct, model: P2GTransferModel, dt: float,
                     grid_mass_grad: wp.array(dtype=wp.float32, ndim=3),
                     grid_momentum_grad: wp.array(dtype=wp.vec3, ndim=3),
                     particle_x_grad: wp.array(dtype=wp.vec3),
                     particle_v_grad: wp.array(dtype=wp.vec3),
                     particle_C_grad: wp.array(dtype=wp.mat33),
                     particle_stress_grad: wp.array(dtype=wp.mat33)):
    p = wp.tid()
    if state.particle_selection[p] != 0:
        return
    grid_pos = state.particle_x[p] * model.inv_dx
    base_x = wp.int(grid_pos[0] - 0.5)
    base_y = wp.int(grid_pos[1] - 0.5)
    base_z = wp.int(grid_pos[2] - 0.5)
    fx = grid_pos - wp.vec3(wp.float(base_x), wp.float(base_y), wp.float(base_z))
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(wp.cw_mul(wa, wa) * 0.5,
                 wp.vec3(0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
                 wp.cw_mul(wc, wc) * 0.5)
    dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
    C = state.particle_C[p]
    Ceff = (1.0 - model.rpic_damping) * C + model.rpic_damping / 2.0 * (C - wp.transpose(C))
    if model.rpic_damping < -0.001:
        Ceff = wp.mat33(0.0)
    stress = state.particle_stress[p]
    mass = state.particle_mass[p]
    volume = state.particle_vol[p]
    velocity = state.particle_v[p]
    gx = wp.vec3(0.0)
    gv = wp.vec3(0.0)
    gCeff = wp.mat33(0.0)
    gstress = wp.mat33(0.0)
    inv_dx2 = model.inv_dx * model.inv_dx
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ix = wp.clamp(base_x + i, 0, model.grid_dim_x - 1)
                iy = wp.clamp(base_y + j, 0, model.grid_dim_y - 1)
                iz = wp.clamp(base_z + k, 0, model.grid_dim_z - 1)
                gm = grid_mass_grad[ix, iy, iz]
                gp = grid_momentum_grad[ix, iy, iz]
                wi = w[0, i]
                wj = w[1, j]
                wk = w[2, k]
                di = dw[0, i]
                dj = dw[1, j]
                dk = dw[2, k]
                weight = wi * wj * wk
                gradw = model.inv_dx * wp.vec3(di * wj * wk,
                                                 wi * dj * wk,
                                                 wi * wj * dk)
                d2i = 1.0
                d2j = 1.0
                d2k = 1.0
                if i == 1:
                    d2i = -2.0
                if j == 1:
                    d2j = -2.0
                if k == 1:
                    d2k = -2.0
                hessian = inv_dx2 * wp.mat33(
                    d2i * wj * wk, di * dj * wk, di * wj * dk,
                    di * dj * wk, wi * d2j * wk, wi * dj * dk,
                    di * wj * dk, wi * dj * dk, wi * wj * d2k)
                dpos = (wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx) * model.dx
                transported = velocity + Ceff * dpos
                gx = gx + mass * (gm + wp.dot(gp, transported)) * gradw
                gx = gx - weight * mass * wp.transpose(Ceff) * gp
                gx = gx - dt * volume * hessian * (wp.transpose(stress) * gp)
                gv = gv + weight * mass * gp
                gCeff = gCeff + weight * mass * wp.outer(gp, dpos)
                gstress = gstress - dt * volume * wp.outer(gp, gradw)
    gC = ((1.0 - model.rpic_damping / 2.0) * gCeff -
          model.rpic_damping / 2.0 * wp.transpose(gCeff))
    if model.rpic_damping < -0.001:
        gC = wp.mat33(0.0)
    particle_x_grad[p] = particle_x_grad[p] + gx
    particle_v_grad[p] = particle_v_grad[p] + gv
    particle_C_grad[p] = particle_C_grad[p] + gC
    particle_stress_grad[p] = particle_stress_grad[p] + gstress


def launch(stage, model, dt, particle_count, grid_shape):
    import os
    if os.environ.get('FLOW_P2G_PRIMAL', 'fixed') == 'fp64_atomic':
        accurate_primal(stage, model, dt, particle_count, grid_shape)
        sorted_grid = None
    else:
        sorted_grid = fixed_primal(stage, model, dt, particle_count, record_tape=False)
    tape = wp.context.runtime.tape
    if tape is None:
        return sorted_grid
    small_model = transfer_model(model)

    def backward_particle():
        wp.launch(p2g_particle_vjp, particle_count,
                  inputs=[stage, small_model, dt, stage.grid_m.grad, stage.grid_v_in.grad,
                          stage.particle_x.grad, stage.particle_v.grad,
                          stage.particle_C.grad, stage.particle_stress.grad],
                  device=stage.particle_x.device, record_tape=False)

    tape.record_func(backward_particle,
                     [stage.grid_m, stage.grid_v_in, stage.particle_x,
                      stage.particle_v, stage.particle_C, stage.particle_stress])
    return sorted_grid

# Optional accurate accumulation: the pointwise contribution is unchanged, but
# atomics accumulate in FP64 before one cast to the solver's FP32 grid. This
# avoids sorting millions of stencil keys in every untaped FD forward pass.
@wp.kernel(enable_backward=False)
def p2g_accumulate64(state: MPMStateStruct, model: P2GTransferModel, dt: float,
                     mass: wp.array(dtype=wp.float64, ndim=3),
                     momentum: wp.array(dtype=wp.float64, ndim=4)):
    p=wp.tid()
    if state.particle_selection[p]!=0:return
    gp=state.particle_x[p]*model.inv_dx
    bx=wp.int(gp[0]-.5);by=wp.int(gp[1]-.5);bz=wp.int(gp[2]-.5)
    fx=gp-wp.vec3(wp.float(bx),wp.float(by),wp.float(bz))
    wa=wp.vec3(1.5)-fx;wb=fx-wp.vec3(1.);wc=fx-wp.vec3(.5)
    w=wp.mat33(wp.cw_mul(wa,wa)*.5,wp.vec3(.75)-wp.cw_mul(wb,wb),wp.cw_mul(wc,wc)*.5)
    dw=wp.mat33(fx-wp.vec3(1.5),-2.*(fx-wp.vec3(1.)),fx-wp.vec3(.5))
    C=state.particle_C[p]
    C=(1.-model.rpic_damping)*C+model.rpic_damping/2.*(C-wp.transpose(C))
    if model.rpic_damping<-.001:C=wp.mat33(0.)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ix=wp.clamp(bx+i,0,model.grid_dim_x-1);iy=wp.clamp(by+j,0,model.grid_dim_y-1);iz=wp.clamp(bz+k,0,model.grid_dim_z-1)
                weight=w[0,i]*w[1,j]*w[2,k]
                dpos=(wp.vec3(wp.float(i),wp.float(j),wp.float(k))-fx)*model.dx
                gw=model.inv_dx*wp.vec3(dw[0,i]*w[1,j]*w[2,k],w[0,i]*dw[1,j]*w[2,k],w[0,i]*w[1,j]*dw[2,k])
                elastic_force=-state.particle_vol[p]*state.particle_stress[p]*gw
                contribution=weight*state.particle_mass[p]*(state.particle_v[p]+C*dpos)+dt*elastic_force
                wp.atomic_add(mass,ix,iy,iz,wp.float64(weight*state.particle_mass[p]))
                for axis in range(3):wp.atomic_add(momentum,ix,iy,iz,axis,wp.float64(contribution[axis]))

@wp.kernel(enable_backward=False)
def cast_grid64(mass:wp.array(dtype=wp.float64,ndim=3),momentum:wp.array(dtype=wp.float64,ndim=4),out_mass:wp.array(dtype=float,ndim=3),out_momentum:wp.array(dtype=wp.vec3,ndim=3)):
    i,j,k=wp.tid()
    out_mass[i,j,k]=wp.float32(mass[i,j,k])
    out_momentum[i,j,k]=wp.vec3(wp.float32(momentum[i,j,k,0]),wp.float32(momentum[i,j,k,1]),wp.float32(momentum[i,j,k,2]))

_workspace={}
def accurate_primal(stage,model,dt,particle_count,shape):
    device=stage.particle_x.device;key=(str(device),tuple(shape))
    if key not in _workspace:
        _workspace[key]=(wp.zeros(shape,dtype=wp.float64,device=device),wp.zeros((*shape,3),dtype=wp.float64,device=device))
    mass,momentum=_workspace[key];mass.zero_();momentum.zero_()
    small=transfer_model(model)
    wp.launch(p2g_accumulate64,particle_count,inputs=[stage,small,dt,mass,momentum],device=device,record_tape=False)
    wp.launch(cast_grid64,shape,inputs=[mass,momentum,stage.grid_m,stage.grid_v_in],device=device,record_tape=False)
