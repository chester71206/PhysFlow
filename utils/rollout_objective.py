"""Explicit objective selection; default weights reproduce historical losses."""
import math


class RolloutObjective:
    def __init__(self, config):
        self.mode = config.get('training_objective', 'flow_geometry')
        if self.mode not in ('flow_geometry', 'flow', 'geometry'):
            raise ValueError('training_objective must be flow_geometry, flow or geometry')
        self.flow_weight = float(config.get('flow_loss_weight', 1.))
        self.geometry_weight = float(config.get('geometry_loss_weight', 10.))
        if not all(math.isfinite(x) and x >= 0 for x in (self.flow_weight, self.geometry_weight)):
            raise ValueError('Objective weights must be finite and nonnegative')
        if not self.needs_flow and not self.needs_geometry:
            raise ValueError('At least one selected objective weight must be positive')

    @property
    def needs_flow(self):
        return self.mode != 'geometry' and self.flow_weight > 0

    @property
    def needs_geometry(self):
        return self.mode != 'flow' and self.geometry_weight > 0

    def terms(self, flow_terms, geometry_losses):
        import torch
        result = {name: value * self.flow_weight for name, value in flow_terms.items()
                  if name.startswith('loss_')} if self.needs_flow else {}
        if self.needs_geometry and geometry_losses:
            result['loss_geo'] = torch.stack(geometry_losses).mean() * self.geometry_weight
        if not result:
            raise ValueError('No usable supervision for the selected objective')
        return result
