"""VJP of the polar rotation using singular-value sums, stable at repeated roots.

Forward is exactly the reference's transposed proper polar rotation. The VJP
is valid for nonsingular orientation-preserving F (physical non-inverted MPM).
It avoids differentiating individual SVD vectors, whose Jacobians are singular
when two singular values coincide although the polar rotation is smooth.
"""
import warp as wp
from mpm_solver_warp.mpm_solver_warp import compute_R_from_F

@wp.kernel(enable_backward=False)
def polar_vjp(Fs: wp.array(dtype=wp.mat33), seeds: wp.array(dtype=wp.mat33), grads: wp.array(dtype=wp.mat33)):
    p=wp.tid()
    U=wp.mat33(0.0);V=wp.mat33(0.0);s=wp.vec3(0.0)
    wp.svd3(Fs[p],U,s,V)
    if wp.determinant(U)<0.0:
        for i in range(3): U[i,2]=-U[i,2]
        s[2]=-s[2]
    if wp.determinant(V)<0.0:
        for i in range(3): V[i,2]=-V[i,2]
        s[2]=-s[2]
    H=wp.transpose(U)*wp.transpose(seeds[p])*V
    K=wp.mat33(0.0)
    for i in range(3):
        for j in range(3):
            if i!=j:
                denom=s[i]+s[j]
                if wp.abs(denom)>1.e-8: K[i,j]=(H[i,j]-H[j,i])/denom
    grads[p]=grads[p]+U*K*wp.transpose(V)

def launch(state,model,n,device):
    wp.launch(compute_R_from_F,n,inputs=[state,model],device=device,record_tape=False)
    tape=wp.context.runtime.tape
    if tape is not None:
        def reverse():
            wp.launch(polar_vjp,n,inputs=[state.particle_F_trial,state.particle_R.grad,state.particle_F_trial.grad],device=device,record_tape=False)
        tape.record_func(reverse,[state.particle_R,state.particle_F_trial])
