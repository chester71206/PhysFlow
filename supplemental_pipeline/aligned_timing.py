"""Explicit absolute-time schedules for the isolated cat timing experiment."""
import math


def keyframe_steps(times, dt, n_key_frames):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('substep_dt must be positive and finite')
    if len(times) != n_key_frames + 1:
        raise ValueError('reference_frame_times must contain n_key_frame + 1 timestamps')
    if any(not math.isfinite(t) for t in times) or times[0] != 0:
        raise ValueError('Reference times must be finite and start at zero')
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Reference times must increase strictly')
    steps = [round(t / dt) for t in times]
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError('Each reference interval must contain at least one substep')
    return steps


def steps_to_target(target_step, current_time, dt):
    remaining = target_step - round(current_time / dt)
    if remaining <= 0:
        raise ValueError('Simulation has already reached or passed the target time')
    return remaining
