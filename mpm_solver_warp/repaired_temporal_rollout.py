"""Checkpointed rollout with analytic particle-owned P2G VJP."""
import warp as wp

from mpm_solver_warp.temporal_rollout import TemporalRollout, view, fresh, zeros
from mpm_solver_warp.temporal_kernels import (
    StressOutput, ParticleOutput, stress_out, grid_scale_out, boundary_out, g2p_out,
)
from mpm_solver_warp.mpm_solver_warp import grid_normalization_and_gravity
from mpm_solver_warp.analytic_p2g_adjoint import launch as fixed_p2g
from mpm_solver_warp.deterministic_g2p_adjoint import launch as fixed_g2p_adjoint


class RepairedTemporalRollout(TemporalRollout):
    def __init__(self, solver):
        import os
        super().__init__(solver)
        self.analytic_g2p = os.environ.get('FLOW_ANALYTIC_G2P', '1') == '1'
    def observe(self, state, model):
        from mpm_solver_warp.mpm_solver_warp import compute_cov_from_F
        from mpm_solver_warp.polar_adjoint import launch as polar
        output = view(state)
        if not model.update_cov_with_F:
            output.particle_cov = fresh(state.particle_cov)
            wp.launch(compute_cov_from_F, self.n, inputs=[output, model], device=self.device)
        output.particle_R = fresh(state.particle_R)
        polar(output, model, self.n, self.device)
        return output.particle_x, output.particle_cov, output.particle_R

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
