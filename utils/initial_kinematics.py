"""Explicit velocity/gravity coordinate conventions for aligned rollouts.

Material coefficients and boundary coordinates remain in MPM units. World
vectors are rotated/scaled with the same transform as particle positions.
"""
import math
import torch
import warp as wp


def configure_initial_kinematics(solver, config, rotation_matrices, scale):
    if not math.isfinite(float(scale)) or float(scale) <= 0:
        raise ValueError('Spatial scale must be positive and finite')
    def vector(name, default):
        values = config.get(name, default)
        if not isinstance(values, (list, tuple)) or len(values) != 3 or not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f'{name} must be a finite three-vector')
        result = torch.tensor(values, dtype=torch.float32, device=str(solver.mpm_state.particle_x.device))
        space = config.get(name + '_space', 'mpm')
        if space not in ('mpm', 'world'):
            raise ValueError(f'{name}_space must be mpm or world')
        if space == 'world':
            for rotation in rotation_matrices:
                result = result @ rotation.to(result).T
            result = result * float(scale)
        return result

    gravity = vector('g', [0., -9.8, 0.])
    velocity = vector('initial_velocity', [0., 0., 0.])
    solver.mpm_model.gravitational_accelaration = wp.vec3(*gravity.tolist())
    solver.set_initial_velocity_from_torch(velocity.expand(solver.n_particles, 3).contiguous())
    return dict(gravity_mpm=gravity.tolist(), initial_velocity_mpm=velocity.tolist(),
                gravity_space=config.get('g_space', 'mpm'),
                initial_velocity_space=config.get('initial_velocity_space', 'mpm'),
                scale_origin=float(scale), material_space='mpm', boundary_space='mpm')
