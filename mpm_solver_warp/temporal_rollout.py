"""Immutable MPM substeps and a segmented, recomputed temporal adjoint.

This module is isolated from the reference/verified solver. Arrays are never
overwritten while their primal values are needed by a tape. Checkpoints contain
particle state only; each reverse segment recreates its grids and intermediates.
"""
from dataclasses import dataclass
import numpy as np
import torch
import warp as wp

from mpm_solver_warp.mpm_solver_warp import (
    compute_mu_lam_from_E_nu, p2g_apic_with_stress,
    grid_normalization_and_gravity, compute_cov_from_F, compute_R_from_F,
)
from mpm_solver_warp.temporal_kernels import (
    stress_out, g2p_out, grid_scale_out, boundary_out, StressOutput, ParticleOutput, TransferModel,
)


def view(source):
    result = source._cls()
    for name in source._cls.vars:
        value = getattr(source, name)
        setattr(result, name, getattr(value, 'value', value))
    return result


def fresh(array):
    return zeros(array.shape, array.dtype, array.device)


def zeros(shape, dtype, device):
    # Warp 0.10 cudaMalloc/cudaFree synchronizes on each fresh allocation. Use
    # Torch's caching allocator, while retaining one distinct live allocation
    # per taped primal. from_torch owns a reference to tensor and its grad.
    components = getattr(dtype, '_shape_', ())
    tensor = torch.zeros((*shape, *components), dtype=torch.float32, device=str(device))
    gradient = wp.from_torch(torch.zeros_like(tensor), dtype=dtype, requires_grad=False)
    # Explicit dtype is required: Warp 0.10 otherwise wraps matrix/vector
    # tensor.grad as a scalar array and rejects the incompatible descriptor.
    return wp.from_torch(tensor, dtype=dtype, requires_grad=True, grad=gradient)


@dataclass
class RolloutRecord:
    checkpoints: dict
    steps: tuple
    dt: float
    interval: int
    parameters: tuple


class TemporalRollout:
    def __init__(self, solver):
        self.device = solver.mpm_state.particle_x.device
        self.n = solver.n_particles
        if solver.mpm_model.material not in (0, 4, 5):
            raise ValueError('Temporal rollout currently validates elastic, plasticine, Newtonian only')
        if solver.pre_p2g_operations or solver.particle_velocity_modifiers or any(x is not None for x in solver.modify_bc):
            raise ValueError('Mutable controllers are not part of this temporal adjoint')
        hardening = solver.mpm_model.hardening
        if getattr(hardening, 'value', hardening) != 0 or np.any(solver.mpm_model.softening.numpy() != 0):
            raise ValueError('Hardening/softening require an additional material-state adjoint')
        if solver.mpm_model.E.size != 1 or solver.mpm_model.nu.size != 1:
            raise ValueError('Temporal correctness work is restricted to global material scalars')
        self.boundaries = []
        for kernel, param in zip(solver.grid_postprocess, solver.collider_params):
            name = kernel.func.__qualname__
            if name == 'MPM_Simulator_WARP.add_bounding_box.<locals>.collide':
                kind = 0
            elif name == 'MPM_Simulator_WARP.add_surface_collider.<locals>.collide':
                kind = 1
            else:
                raise ValueError(f'Unvalidated boundary in temporal rollout: {name}')
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
        self.transfer_model = TransferModel()
        for name in self.transfer_model._cls.vars:
            setattr(self.transfer_model, name, getattr(self.model_template, name))
        self.max_segment_steps = 0

    def checkpoint(self, state):
        return {name: getattr(state, name).numpy().copy() for name in self.dynamic}

    def restore(self, checkpoint):
        state = view(self.template)
        for name, data in checkpoint.items():
            dtype = getattr(self.template, name).dtype
            setattr(state, name, wp.from_numpy(data, dtype=dtype, device=self.device, requires_grad=True))
        return state

    def model(self, parameters):
        model = view(self.model_template)
        for name, value in zip(('E', 'nu', 'yield_stress'), parameters):
            setattr(model, name, wp.array([float(value)], dtype=float, device=self.device, requires_grad=True))
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
        wp.launch(stress_out, self.n, inputs=[state, stress, model], device=self.device)
        wp.launch(p2g_apic_with_stress, self.n, inputs=[stage, model, dt], device=self.device)
        wp.launch(grid_normalization_and_gravity, self.grid_shape, inputs=[stage, model, dt], device=self.device)
        if model.grid_v_damping_scale < 1.0:
            following = view(stage)
            following.grid_v_out = fresh(stage.grid_v_out)
            wp.launch(grid_scale_out, self.grid_shape,
                      inputs=[stage.grid_v_out, following.grid_v_out, model.grid_v_damping_scale], device=self.device)
            stage = following
        for kind, param in self.boundaries:
            following = view(stage)
            following.grid_v_out = fresh(stage.grid_v_out)
            wp.launch(boundary_out, self.grid_shape,
                      inputs=[stage.grid_v_out, following.grid_v_out, model, param, kind, time], device=self.device)
            stage = following
        out = view(stage)
        for name in ('particle_x', 'particle_v', 'particle_C', 'particle_F_trial', 'particle_F_N_trial'):
            setattr(out, name, fresh(getattr(state, name)))
        if model.update_cov_with_F:
            out.particle_cov = fresh(state.particle_cov)
        particle_output = ParticleOutput()
        for name in particle_output._cls.vars:
            setattr(particle_output, name, getattr(out, name))
        wp.launch(g2p_out, self.n, inputs=[stage, particle_output, self.transfer_model, dt], device=self.device)
        return out

    def observe(self, state, model):
        output = view(state)
        if not model.update_cov_with_F:
            output.particle_cov = fresh(state.particle_cov)
            wp.launch(compute_cov_from_F, self.n, inputs=[output, model], device=self.device)
        output.particle_R = fresh(state.particle_R)
        wp.launch(compute_R_from_F, self.n, inputs=[output, model], device=self.device)
        return (output.particle_x, output.particle_cov, output.particle_R)

    def forward(self, parameters, steps, dt, interval=32):
        steps = tuple(int(x) for x in steps)
        if not steps or steps[0] < 1 or any(b <= a for a, b in zip(steps, steps[1:])):
            raise ValueError('Observation steps must be positive and strictly increasing')
        if interval < 1 or dt <= 0:
            raise ValueError('Positive checkpoint interval and timestep required')
        state = self.restore(self.initial)
        model = self.model(parameters)
        wp.launch(compute_mu_lam_from_E_nu, self.n, inputs=[state, model], device=self.device)
        checkpoints = {0: self.initial}
        outputs = [[], [], []]
        selected = set(steps)
        for t in range(1, steps[-1]+1):
            state = self.step(state, model, dt, (t-1)*dt)
            if t in selected:
                for rows, array in zip(outputs, self.observe(state, model)):
                    rows.append(wp.to_torch(array).detach().clone())
            if t % interval == 0 and t < steps[-1]:
                checkpoints[t] = self.checkpoint(state)
        record = RolloutRecord(checkpoints, steps, dt, interval, tuple(parameters))
        return tuple(torch.stack(rows) for rows in outputs), record

    def backward(self, record, cotangents):
        seeds = {step: tuple(g[i] if g is not None else None for g in cotangents)
                 for i, step in enumerate(record.steps)}
        boundaries = sorted(record.checkpoints) + [record.steps[-1]]
        carry = {}
        gradients = np.zeros(3, dtype=np.float64)
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
            # A state at a segment end can receive both an observation seed and
            # the later segment's state cotangent. Sum them before one backward.
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
            # Warp 0.10 structs cache array descriptors, including the grad
            # pointer. Tape.backward(grads=...) replaces array.grad and leaves
            # those descriptors pointing at freed storage. Seed the existing
            # buffers instead; their pointers remain valid for every launch.
            for array, seed in warp_seeds.items():
                wp.copy(array.grad, seed)
            tape.backward()
            for i, name in enumerate(('E', 'nu', 'yield_stress')):
                gradients[i] += float(getattr(model, name).grad.numpy()[0])
            carry = {name: wp.to_torch(getattr(initial, name).grad).detach().clone()
                     for name in self.dynamic}
            # Release every taped primal before allocating the next segment.
            tape.reset()
            del tape, observations, combined, warp_seeds, initial, state, model
        return gradients

    def apply(self, E, nu, yield_stress, steps, dt, interval=32):
        return _TemporalFunction.apply(E, nu, yield_stress, self, tuple(steps), dt, interval)


class _TemporalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, E, nu, yield_stress, engine, steps, dt, interval):
        ctx.set_materialize_grads(False)
        ctx.engine = engine
        ctx.templates = (E, nu, yield_stress)
        outputs, ctx.record = engine.forward([float(p.detach()) for p in ctx.templates], steps, dt, interval)
        return outputs

    @staticmethod
    def backward(ctx, *cotangents):
        grads = ctx.engine.backward(ctx.record, cotangents)
        return (*[torch.full_like(p, float(g)) for p, g in zip(ctx.templates, grads)], None, None, None, None)
