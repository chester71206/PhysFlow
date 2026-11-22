"""Checkpointed reverse AD for elastic MPM with scheduled moving grid cuboids.

The original Warp controller mutates its center after each active substep.
Here the same center is reconstructed from absolute time, so reverse segments
replay it without inheriting mutable controller state. The cuboid schedule is
fixed; material derivatives pass through the grid velocity projection.
"""
import copy

import warp as wp

from mpm_solver_warp.repaired_temporal_rollout import RepairedTemporalRollout
from mpm_solver_warp.temporal_rollout import view, fresh, zeros
from mpm_solver_warp.temporal_kernels import (
    StressOutput, ParticleOutput, stress_out, grid_scale_out, boundary_out, g2p_out,
)
from mpm_solver_warp.mpm_solver_warp import MPMModelStruct, grid_normalization_and_gravity
from mpm_solver_warp.analytic_p2g_adjoint import launch as fixed_p2g
from mpm_solver_warp.deterministic_g2p_adjoint import launch as fixed_g2p_adjoint
from mpm_solver_warp.warp_utils import Dirichlet_collider


@wp.kernel
def moving_cuboid_out(before: wp.array(dtype=wp.vec3, ndim=3),
                      after: wp.array(dtype=wp.vec3, ndim=3),
                      model: MPMModelStruct, param: Dirichlet_collider,
                      center: wp.vec3, time: float):
    i, j, k = wp.tid()
    v = before[i, j, k]
    if time >= param.start_time and time < param.end_time:
        offset = wp.vec3(float(i) * model.dx - center[0],
                         float(j) * model.dx - center[1],
                         float(k) * model.dx - center[2])
        if (wp.abs(offset[0]) < param.size[0] and
            wp.abs(offset[1]) < param.size[1] and
            wp.abs(offset[2]) < param.size[2]):
            v = param.velocity
    after[i, j, k] = v


class MovingCuboidTemporalRollout(RepairedTemporalRollout):
    """RepairedTemporalRollout plus deterministic moving-cuboid projections."""

    def __init__(self, solver):
        static = []
        moving = []
        for kernel, param, modify in zip(solver.grid_postprocess,
                                         solver.collider_params, solver.modify_bc):
            name = kernel.func.__qualname__
            if name == 'MPM_Simulator_WARP.set_velocity_on_cuboid.<locals>.collide':
                if modify is None or param.reset != 0:
                    raise ValueError('Only non-reset moving cuboids are validated')
                moving.append(view(param))
            elif modify is None:
                if moving:
                    raise ValueError('Static boundary after moving cuboid changes the application order')
                static.append((kernel, param, modify))
            else:
                raise ValueError(f'Unvalidated mutable boundary: {name}')
        if not moving:
            raise ValueError('No moving cuboid boundary found')
        restricted = copy.copy(solver)
        restricted.grid_postprocess = [item[0] for item in static]
        restricted.collider_params = [item[1] for item in static]
        restricted.modify_bc = [item[2] for item in static]
        super().__init__(restricted)
        self.moving = moving
        self._controller_cache = {}

    def _controller_at(self, step_index, dt):
        """Replay the original float32 point update and Python time increments."""
        key = float(dt)
        if key not in self._controller_cache:
            self._controller_cache[key] = dict(times=[], centers=[], time=0.,
                points=[wp.vec3(p.point[0], p.point[1], p.point[2]) for p in self.moving])
        cache = self._controller_cache[key]
        while len(cache['times']) <= step_index:
            time = cache['time']
            cache['times'].append(time)
            cache['centers'].append(tuple(cache['points']))
            next_points=[]
            for point,param in zip(cache['points'],self.moving):
                if time >= param.start_time and time < param.end_time:
                    point=wp.vec3(point[0]+dt*param.velocity[0],
                                  point[1]+dt*param.velocity[1],
                                  point[2]+dt*param.velocity[2])
                next_points.append(point)
            cache['points']=next_points
            cache['time']=time+dt
        return cache['times'][step_index], cache['centers'][step_index]

    def step(self, state, model, dt, time):
        stage = view(state)
        stage.particle_F = fresh(state.particle_F)
        stage.particle_stress = fresh(state.particle_stress)
        stage.grid_m = zeros(self.grid_shape, wp.float32, self.device)
        stage.grid_v_in = zeros(self.grid_shape, wp.vec3, self.device)
        stage.grid_v_out = zeros(self.grid_shape, wp.vec3, self.device)
        stress = StressOutput()
        stress.particle_F, stress.particle_stress = stage.particle_F, stage.particle_stress
        wp.launch(stress_out, self.n, inputs=[state, stress, model], device=self.device)
        sorted_grid = fixed_p2g(stage, model, dt, self.n, self.grid_shape)
        wp.launch(grid_normalization_and_gravity, self.grid_shape,
                  inputs=[stage, model, dt], device=self.device)
        if model.grid_v_damping_scale < 1.0:
            following = view(stage)
            following.grid_v_out = fresh(stage.grid_v_out)
            wp.launch(grid_scale_out, self.grid_shape,
                      inputs=[stage.grid_v_out, following.grid_v_out,
                              model.grid_v_damping_scale], device=self.device)
            stage = following
        for kind, param in self.boundaries:
            following = view(stage)
            following.grid_v_out = fresh(stage.grid_v_out)
            wp.launch(boundary_out, self.grid_shape,
                      inputs=[stage.grid_v_out, following.grid_v_out,
                              model, param, kind, time], device=self.device)
            stage = following
        controller_time, centers = self._controller_at(round(time / dt), dt)
        for param,center in zip(self.moving,centers):
            following = view(stage)
            following.grid_v_out = fresh(stage.grid_v_out)
            wp.launch(moving_cuboid_out, self.grid_shape,
                      inputs=[stage.grid_v_out, following.grid_v_out,
                              model, param, center, controller_time], device=self.device)
            stage = following
        out = view(stage)
        for name in ('particle_x', 'particle_v', 'particle_C',
                     'particle_F_trial', 'particle_F_N_trial'):
            setattr(out, name, fresh(getattr(state, name)))
        if model.update_cov_with_F:
            out.particle_cov = fresh(state.particle_cov)
        particle_output = ParticleOutput()
        for name in particle_output._cls.vars:
            setattr(particle_output, name, getattr(out, name))
        if wp.context.runtime.tape is None:
            wp.launch(g2p_out, self.n,
                      inputs=[stage, particle_output, self.transfer_model, dt], device=self.device)
        elif self.analytic_g2p:
            from mpm_solver_warp.analytic_g2p_adjoint import launch as analytic_g2p
            analytic_g2p(stage, particle_output, self.transfer_model, model, dt, self.n, sorted_grid)
        else:
            fixed_g2p_adjoint(stage, particle_output, self.transfer_model, model, dt, self.n)
        return out
