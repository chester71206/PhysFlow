"""Particle-owned analytic G2P VJP, with deterministic grid gradient reduction."""
import os,torch,warp as wp
from mpm_solver_warp.temporal_kernels import ParticleOutput,TransferModel,g2p_out
from mpm_solver_warp.deterministic_p2g_diagnostic import contribution_keys
from mpm_solver_warp.deterministic_g2p_adjoint import reduce_local_grid_adjoint

@wp.struct
class Input:
    x: wp.array(dtype=wp.vec3)
    F: wp.array(dtype=wp.mat33)
    Fn: wp.array(dtype=wp.mat33)
    cov: wp.array(dtype=float)
    selection: wp.array(dtype=int)
    grid: wp.array(dtype=wp.vec3,ndim=3)

@wp.struct
class InputGrad:
    x: wp.array(dtype=wp.vec3)
    F: wp.array(dtype=wp.mat33)
    Fn: wp.array(dtype=wp.mat33)
    cov: wp.array(dtype=float)

@wp.func
def contraction(A:wp.mat33,B:wp.mat33):
    v=float(0.)
    for i in range(3):
        for j in range(3):v=v+A[i,j]*B[i,j]
    return v

@wp.kernel(enable_backward=False)
def particle_vjp(state:Input,seed:ParticleOutput,grad:InputGrad,model:TransferModel,dt:float,local:wp.array(dtype=wp.vec3)):
    p=wp.tid()
    for q in range(27):local[p*27+q]=wp.vec3(0.)
    if state.selection[p]!=0:return
    gp=state.x[p]*model.inv_dx
    bx=wp.int(gp[0]-.5);by=wp.int(gp[1]-.5);bz=wp.int(gp[2]-.5)
    fx=gp-wp.vec3(wp.float(bx),wp.float(by),wp.float(bz))
    wa=wp.vec3(1.5)-fx;wb=fx-wp.vec3(1.);wc=fx-wp.vec3(.5)
    w=wp.mat33(wp.cw_mul(wa,wa)*.5,wp.vec3(.75)-wp.cw_mul(wb,wb),wp.cw_mul(wc,wc)*.5)
    dw=wp.mat33(fx-wp.vec3(1.5),-2.*(fx-wp.vec3(1.)),fx-wp.vec3(.5))
    A=wp.mat33(0.)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u=state.grid[wp.clamp(bx+i,0,model.grid_dim_x-1),wp.clamp(by+j,0,model.grid_dim_y-1),wp.clamp(bz+k,0,model.grid_dim_z-1)]
                gw=model.inv_dx*wp.vec3(dw[0,i]*w[1,j]*w[2,k],w[0,i]*dw[1,j]*w[2,k],w[0,i]*w[1,j]*dw[2,k])
                A=A+wp.outer(u,gw)
    B=wp.mat33(1.,0.,0.,0.,1.,0.,0.,0.,1.)+dt*A
    gx=seed.particle_x[p]
    gv=seed.particle_v[p]+dt*gx
    gC=seed.particle_C[p]
    gF=seed.particle_F_trial[p];gFn=seed.particle_F_N_trial[p]
    gA=dt*(gF*wp.transpose(state.F[p])+gFn*wp.transpose(state.Fn[p]))
    if model.update_cov_with_F:
        cov=wp.mat33(state.cov[6*p],state.cov[6*p+1],state.cov[6*p+2],state.cov[6*p+1],state.cov[6*p+3],state.cov[6*p+4],state.cov[6*p+2],state.cov[6*p+4],state.cov[6*p+5])
        gc=wp.mat33(seed.particle_cov[6*p],seed.particle_cov[6*p+1],seed.particle_cov[6*p+2],0.,seed.particle_cov[6*p+3],seed.particle_cov[6*p+4],0.,0.,seed.particle_cov[6*p+5])
        gA=gA+dt*(gc*wp.transpose(cov)+wp.transpose(gc)*cov)
        H=gc+dt*(wp.transpose(A)*gc+gc*A)
        grad.cov[6*p]=grad.cov[6*p]+H[0,0]
        grad.cov[6*p+1]=grad.cov[6*p+1]+H[0,1]+H[1,0]
        grad.cov[6*p+2]=grad.cov[6*p+2]+H[0,2]+H[2,0]
        grad.cov[6*p+3]=grad.cov[6*p+3]+H[1,1]
        grad.cov[6*p+4]=grad.cov[6*p+4]+H[1,2]+H[2,1]
        grad.cov[6*p+5]=grad.cov[6*p+5]+H[2,2]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u=state.grid[wp.clamp(bx+i,0,model.grid_dim_x-1),wp.clamp(by+j,0,model.grid_dim_y-1),wp.clamp(bz+k,0,model.grid_dim_z-1)]
                wi=w[0,i];wj=w[1,j];wk=w[2,k];di=dw[0,i];dj=dw[1,j];dk=dw[2,k]
                weight=wi*wj*wk;gw=model.inv_dx*wp.vec3(di*wj*wk,wi*dj*wk,wi*wj*dk)
                hi=float(1.);hj=float(1.);hk=float(1.)
                if i==1:hi=-2.
                if j==1:hj=-2.
                if k==1:hk=-2.
                H=(model.inv_dx*model.inv_dx)*wp.mat33(hi*wj*wk,di*dj*wk,di*wj*dk,di*dj*wk,wi*hj*wk,wi*dj*dk,di*wj*dk,wi*dj*dk,wi*wj*hk)
                r=wp.vec3(wp.float(i),wp.float(j),wp.float(k))-fx
                gx=gx+wp.dot(gv,u)*gw+4.*model.inv_dx*(contraction(gC,wp.outer(u,r))*gw-weight*model.inv_dx*wp.transpose(gC)*u)+H*wp.transpose(gA)*u
                local[p*27+i*9+j*3+k]=weight*gv+4.*model.inv_dx*weight*gC*r+gA*gw
    grad.x[p]=grad.x[p]+gx
    grad.F[p]=grad.F[p]+wp.transpose(B)*gF
    grad.Fn[p]=grad.Fn[p]+wp.transpose(B)*gFn

@wp.kernel(enable_backward=False)
def frozen_vjp(selection:wp.array(dtype=int),before:ParticleOutput,after:ParticleOutput,advect:int):
    p=wp.tid()
    if selection[p]!=0:
        before.particle_x[p]=before.particle_x[p]+after.particle_x[p]
        before.particle_v[p]=before.particle_v[p]+after.particle_v[p]
        before.particle_C[p]=before.particle_C[p]+after.particle_C[p]
        before.particle_F_trial[p]=before.particle_F_trial[p]+after.particle_F_trial[p]
        before.particle_F_N_trial[p]=before.particle_F_N_trial[p]+after.particle_F_N_trial[p]
        if advect:
            for i in range(6):before.particle_cov[6*p+i]=before.particle_cov[6*p+i]+after.particle_cov[6*p+i]

def launch(stage,output,transfer_model,material_model,dt,count,sorted_grid=None):
    device=stage.particle_x.device;n=count*27
    atomic=os.environ.get('FLOW_G2P_REDUCTION','fixed')=='fp64_atomic'
    if sorted_grid is None and not atomic:
        keys=wp.empty(n,dtype=wp.int32,device=device);values=wp.empty(n,dtype=wp.int32,device=device)
        wp.launch(contribution_keys,n,inputs=[stage,material_model,keys,values],device=device,record_tape=False);wp.synchronize_device(device)
        kt=wp.to_torch(keys);order=torch.argsort(kt,stable=True)
        sk=torch.cat([kt[order],torch.tensor([material_model.grid_dim_x*material_model.grid_dim_y*material_model.grid_dim_z],device=str(device),dtype=torch.int32)])
        sv=torch.cat([order.int(),torch.zeros(1,device=str(device),dtype=torch.int32)])
        sorted_grid=(wp.from_torch(sk),wp.from_torch(sv))
    if not atomic:
        keys,values=sorted_grid
        max_segment=int(torch.unique_consecutive(wp.to_torch(keys)[:n],return_counts=True)[1].max())
    local=wp.from_torch(torch.empty(n,3,device=str(device)),dtype=wp.vec3)
    def gradient(array):
        if array.grad is None:array.grad=wp.zeros_like(array)
        return array.grad
    def backward():
        seed=ParticleOutput();before=ParticleOutput()
        for name in seed._cls.vars:
            setattr(seed,name,gradient(getattr(output,name)));setattr(before,name,gradient(getattr(stage,name)))
        state=Input();state.x=stage.particle_x;state.F=stage.particle_F;state.Fn=stage.particle_F_N;state.cov=stage.particle_cov;state.selection=stage.particle_selection;state.grid=stage.grid_v_out
        grad=InputGrad();grad.x=gradient(stage.particle_x);grad.F=gradient(stage.particle_F);grad.Fn=gradient(stage.particle_F_N);grad.cov=gradient(stage.particle_cov)
        wp.launch(particle_vjp,count,inputs=[state,seed,grad,transfer_model,dt,local],device=device,record_tape=False)
        wp.launch(frozen_vjp,count,inputs=[stage.particle_selection,before,seed,transfer_model.update_cov_with_F],device=device,record_tape=False)
        if atomic:
            shape=tuple(stage.grid_m.shape);grid64=workspace64(device,shape);grid64.zero_()
            wp.launch(scatter_grid64,count,inputs=[state,transfer_model,local,grid64],device=device,record_tape=False)
            wp.launch(add_grid64,shape,inputs=[grid64,stage.grid_v_out.grad],device=device,record_tape=False)
        else:
            wp.launch(reduce_local_grid_adjoint,n,inputs=[local,stage.grid_v_out.grad,keys,values,n,max_segment,material_model],device=device,record_tape=False)
    tape=wp.context.runtime.tape
    if tape is None:raise RuntimeError('Analytic G2P VJP must be recorded on tape')
    arrays=[stage.particle_x,stage.particle_v,stage.particle_C,stage.particle_F,stage.particle_F_trial,stage.particle_F_N_trial,stage.particle_cov,stage.grid_v_out]+[getattr(output,k) for k in output._cls.vars]
    tape.record_func(backward,[x for x in arrays if x.grad is not None])
    wp.launch(g2p_out,count,inputs=[stage,output,transfer_model,dt],device=device,record_tape=False)

@wp.kernel(enable_backward=False)
def scatter_grid64(state:Input,model:TransferModel,local:wp.array(dtype=wp.vec3),grid:wp.array(dtype=wp.float64,ndim=4)):
    p=wp.tid()
    if state.selection[p]!=0:return
    gp=state.x[p]*model.inv_dx
    bx=wp.int(gp[0]-.5);by=wp.int(gp[1]-.5);bz=wp.int(gp[2]-.5)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ix=wp.clamp(bx+i,0,model.grid_dim_x-1);iy=wp.clamp(by+j,0,model.grid_dim_y-1);iz=wp.clamp(bz+k,0,model.grid_dim_z-1)
                value=local[p*27+i*9+j*3+k]
                for a in range(3):wp.atomic_add(grid,ix,iy,iz,a,wp.float64(value[a]))

@wp.kernel(enable_backward=False)
def add_grid64(grid:wp.array(dtype=wp.float64,ndim=4),out:wp.array(dtype=wp.vec3,ndim=3)):
    i,j,k=wp.tid()
    out[i,j,k]=out[i,j,k]+wp.vec3(wp.float32(grid[i,j,k,0]),wp.float32(grid[i,j,k,1]),wp.float32(grid[i,j,k,2]))

_grid_workspaces={}
def workspace64(device,shape):
    key=(str(device),tuple(shape))
    if key not in _grid_workspaces:_grid_workspaces[key]=wp.zeros((*shape,3),dtype=wp.float64,device=device)
    return _grid_workspaces[key]
