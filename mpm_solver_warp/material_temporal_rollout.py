"""Experimental immutable temporal MPM path for PAC-NeRF's nonelastic materials.

This lives beside the frozen Letter rollout. A material is admitted for training
only after forward parity and AD/FD checks pass for that material and scene.
"""
import numpy as np
import torch
import warp as wp

from mpm_solver_warp.mpm_solver_warp import (
    MPMModelStruct, MPMStateStruct, compute_mu_lam_from_E_nu,
    grid_normalization_and_gravity, kirchoff_stress_StVK,
    kirchoff_stress_drucker_prager,
)
from mpm_solver_warp.mpm_utils import (
    sand_return_mapping, viscoplasticity_return_mapping_with_StVK_force,
)
from mpm_solver_warp.temporal_kernels import (
    StressOutput, ParticleOutput, plasticine_return_mapping_readonly,
    grid_scale_out, boundary_out, g2p_out,
)
from mpm_solver_warp.temporal_rollout import (
    RolloutRecord, TemporalRollout, fresh, view, zeros,
)
from mpm_solver_warp.repaired_temporal_rollout import RepairedTemporalRollout
from mpm_solver_warp.analytic_p2g_adjoint import launch as fixed_p2g
from mpm_solver_warp.deterministic_g2p_adjoint import launch as fixed_g2p_adjoint


PARAMETERS = ('E', 'nu', 'yield_stress', 'plastic_viscosity', 'friction_angle')


@wp.kernel
def material_stress_out(state: MPMStateStruct, out: StressOutput,
                        model: MPMModelStruct, dt: float):
    p = wp.tid()
    F = state.particle_F[p]
    stress = state.particle_stress[p]
    if state.particle_selection[p] == 0:
        F = state.particle_F_trial[p]
        if model.material == 4:
            F = plasticine_return_mapping_readonly(F, model, p)
        elif model.material == 6:
            F = viscoplasticity_return_mapping_with_StVK_force(F, model, p, dt)
        elif model.material == 2:
            F = sand_return_mapping(F, state, model, p)
        J = wp.determinant(F)
        U = wp.mat33(0.0)
        V = wp.mat33(0.0)
        sig = wp.vec3(0.0)
        wp.svd3(F, U, sig, V)
        if model.material == 2:
            stress = kirchoff_stress_drucker_prager(F, U, V, sig, model.mu[p], model.lam[p])
        else:
            stress = kirchoff_stress_StVK(F, U, V, sig, model.mu[p], model.lam[p])
        stress = (stress + wp.transpose(stress)) / 2.0
    out.particle_F[p] = F
    out.particle_stress[p] = stress


class MaterialTemporalRollout(RepairedTemporalRollout):
    """Five-scalar extension of the frozen checkpointed reverse AD rollout."""

    def __init__(self, solver):
        # Copy only initialization to admit the material without changing the
        # frozen rollout's material gate while Letter is running.
        self.device = solver.mpm_state.particle_x.device
        self.n = solver.n_particles
        if solver.mpm_model.material not in (2, 4, 6):
            raise ValueError('MaterialTemporalRollout covers sand, plasticine, non-Newtonian')
        if solver.pre_p2g_operations or solver.particle_velocity_modifiers or any(
                x is not None for x in solver.modify_bc):
            raise ValueError('Mutable controllers are not part of this temporal adjoint')
        hardening = solver.mpm_model.hardening
        if getattr(hardening, 'value', hardening) != 0 or np.any(solver.mpm_model.softening.numpy() != 0):
            raise ValueError('Hardening/softening require another material-state adjoint')
        if any(getattr(solver.mpm_model, name).size != 1 for name in PARAMETERS):
            raise ValueError('Only global material scalars are supported')
        self.boundaries = []
        for kernel, param in zip(solver.grid_postprocess, solver.collider_params):
            name = kernel.func.__qualname__
            if name == 'MPM_Simulator_WARP.add_bounding_box.<locals>.collide':
                kind = 0
            elif name == 'MPM_Simulator_WARP.add_surface_collider.<locals>.collide':
                kind = 1
            else:
                raise ValueError(f'Unvalidated boundary: {name}')
            self.boundaries.append((kind, view(param)))
        self.template = view(solver.mpm_state)
        for name in self.template._cls.vars:
            value = getattr(self.template, name)
            if isinstance(value, wp.array) and not name.startswith('grid_'):
                setattr(self.template, name, wp.clone(value, requires_grad=False))
        self.model_template = view(solver.mpm_model)
        for name in self.model_template._cls.vars:
            value = getattr(self.model_template, name)
            if isinstance(value, wp.array):
                setattr(self.model_template, name, wp.clone(value, requires_grad=False))
        self.dynamic = ['particle_x', 'particle_v', 'particle_C', 'particle_F_trial',
                        'particle_F_N_trial', 'particle_F', 'particle_stress']
        if self.model_template.update_cov_with_F:
            self.dynamic.append('particle_cov')
        self.initial = self.checkpoint(self.template)
        self.grid_shape = tuple(solver.mpm_state.grid_m.shape)
        from mpm_solver_warp.temporal_kernels import TransferModel
        self.transfer_model = TransferModel()
        for name in self.transfer_model._cls.vars:
            setattr(self.transfer_model, name, getattr(self.model_template, name))
        self.max_segment_steps = 0
        import os
        self.analytic_g2p = os.environ.get('FLOW_ANALYTIC_G2P', '1') == '1'

    def model(self, parameters):
        model = view(self.model_template)
        for name, value in zip(PARAMETERS, parameters):
            setattr(model, name, wp.array([float(value)], dtype=float,
                                           device=self.device, requires_grad=True))
        model.mu = fresh(self.model_template.mu)
        model.lam = fresh(self.model_template.lam)
        return model

    def step(self, state, model, dt, time):
        stage = view(state)
        stage.particle_F = fresh(state.particle_F)
        stage.particle_stress = fresh(state.particle_stress)
        stage.grid_m = zeros(self.grid_shape, wp.float32, self.device)
        stage.grid_v_in = zeros(self.grid_shape, wp.vec3, self.device)
        stage.grid_v_out = zeros(self.grid_shape, wp.vec3, self.device)
        stress = StressOutput()
        stress.particle_F, stress.particle_stress = stage.particle_F, stage.particle_stress
        wp.launch(material_stress_out, self.n, inputs=[state, stress, model, dt], device=self.device)
        if getattr(self, 'diagnostic_hook', None) is not None:
            self.diagnostic_hook('stress', stage, state)
        sorted_grid = fixed_p2g(stage, model, dt, self.n, self.grid_shape)
        if getattr(self, 'diagnostic_hook', None) is not None:
            self.diagnostic_hook('p2g', stage, state)
        wp.launch(grid_normalization_and_gravity, self.grid_shape,
                  inputs=[stage, model, dt], device=self.device)
        if getattr(self, 'diagnostic_hook', None) is not None:
            self.diagnostic_hook('grid', stage, state)
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
        if getattr(self, 'diagnostic_hook', None) is not None:
            self.diagnostic_hook('boundary', stage, state)
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
        if getattr(self, 'diagnostic_hook', None) is not None:
            self.diagnostic_hook('g2p', out, state)
        return out

    def backward(self, record: RolloutRecord, cotangents):
        seeds = {step: tuple(g[i] if g is not None else None for g in cotangents)
                 for i, step in enumerate(record.steps)}
        boundaries = sorted(record.checkpoints) + [record.steps[-1]]
        carry = {}
        gradients = np.zeros(len(PARAMETERS), dtype=np.float64)
        for start, end in reversed(list(zip(boundaries, boundaries[1:]))):
            initial = self.restore(record.checkpoints[start])
            model = self.model(record.parameters)
            state = initial
            tape = wp.Tape()
            observations = {}
            with tape:
                wp.launch(compute_mu_lam_from_E_nu, self.n, inputs=[state, model], device=self.device)
                for t in range(start+1, end+1):
                    state = self.step(state, model, record.dt, (t-1)*record.dt)
                    if t in seeds:
                        observations[t] = self.observe(state, model)
            self.max_segment_steps = max(self.max_segment_steps, end-start)
            combined = {}
            def add(array, tensor):
                if tensor is not None:
                    tensor = tensor.detach().contiguous()
                    combined[array] = combined.get(array, 0) + tensor
            for name, tensor in carry.items():
                add(getattr(state, name), tensor)
            for t, arrays in observations.items():
                for array, gradient in zip(arrays, seeds[t]):
                    add(array, gradient)
            warp_seeds = {a: wp.from_torch(g.contiguous(), dtype=a.dtype) for a, g in combined.items()}
            for array, seed in warp_seeds.items():
                wp.copy(array.grad, seed)
            tape.backward()
            for i, name in enumerate(PARAMETERS):
                gradients[i] += float(getattr(model, name).grad.numpy()[0])
            carry = {name: wp.to_torch(getattr(initial, name).grad).detach().clone()
                     for name in self.dynamic}
            tape.reset()
            del tape, observations, combined, warp_seeds, initial, state, model
        return gradients

    def apply(self, *args):
        return _MaterialFunction.apply(*args, self)


class _MaterialFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        *parameters, steps, dt, interval, engine = args
        ctx.set_materialize_grads(False)
        ctx.engine = engine
        ctx.templates = parameters
        outputs, ctx.record = engine.forward([float(p.detach()) for p in parameters],
                                             steps, dt, interval)
        return outputs

    @staticmethod
    def backward(ctx, *cotangents):
        grads = ctx.engine.backward(ctx.record, cotangents)
        return (*[torch.full_like(p, float(g)) for p, g in zip(ctx.templates, grads)],
                None, None, None, None)
