"""Summarize the checkpointed nine-scene WonderPlay multistart queue."""
import argparse
import csv
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs/pacnerf_wonderplay_multistart_20261122'))
    args = parser.parse_args()
    jobs = read_json(args.root / 'runner_state.json')
    if jobs is None:
        parser.error('runner_state.json missing')
    rows = []
    for job in jobs:
        directory = Path(job['destination'])
        history = []
        gates = []
        for segment in job['segments']:
            part = directory / segment['name']
            for filename in ('gradient_check.json', 'material_ad_check.json'):
                gate = read_json(part / filename)
                if gate is not None:
                    gates.append(gate)
            path = part / 'training.jsonl'
            if path.is_file():
                history.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        history.sort(key=lambda item: item['epoch'])
        anchor = job['anchor']
        final = history[-1]['after'] if history else job['initial']
        errors = {}
        boundaries = []
        for name in job['active_parameters']:
            value, target = float(final[name]), float(anchor[name])
            errors[name] = (abs(value - target) if name in ('nu', 'friction_angle')
                            else abs(value / target - 1) if target else abs(value))
            low, high = ((.01, 89.) if name == 'friction_angle' else
                         ((1-10**-.0001)/2, (1-10**-6)/2) if name == 'nu' else
                         (1e-7, 10**-.4) if job['material'] in ('elastic', 'newtonian') and name == 'E'
                         else (1e-12, 1e4))
            if abs(value - low) <= max(abs(low) * 1e-5, 1e-9) or abs(value - high) <= max(abs(high) * 1e-5, 1e-9):
                boundaries.append(name)
        near = all(errors[name] <= (.01 if name == 'nu' else 5. if name == 'friction_angle' else .1)
                   for name in errors)
        gate = gates[0] if gates else {}
        row = dict(id=job['id'], scene=job['scene'], start=job['start'], material=job['material'],
                   status=job['status'], updates=len(history), initial_loss=(history[0]['loss_before']
                   if history else gate.get('loss')), final_loss=(history[-1]['loss_after'] if history else None),
                   gate_passed=gate.get('passed'), stagewise_passed=gate.get('stagewise_passed'),
                   anchor=anchor, initial=job['initial'], final=final, errors=errors,
                   touched_boundary=boundaries, near_gt=near if history else None,
                   interpretation=('near_GT' if near else 'away_from_GT') if job['status'] == 'complete' else
                   'in_progress' if history else
                   'numerical_or_gate_failure' if job['status'] == 'failed' else job['status'])
        rows.append(row)
    (args.root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (args.root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['id', 'status', 'updates', 'initial_loss',
            'final_loss', 'gate_passed', 'stagewise_passed', 'near_gt', 'touched_boundary', 'errors'])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if key in ('touched_boundary', 'errors') else row[key]
                             for key in writer.fieldnames})
    print(args.root / 'summary.csv')


if __name__ == '__main__':
    main()
