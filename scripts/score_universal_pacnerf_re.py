"""PAC-NeRF material RE in the GT table's physical units where conversion is known."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/time_aligned_universal_batch_20261018'
TRUTH=json.loads((ROOT/'data/pacnerf/groundtruth.json').read_text())['scenes']


def score(scene,params,scale):
    gt=TRUTH[scene]['parameters']
    e=float(params.get('E',0));nu=float(params.get('nu',0))
    physical={}
    if scene in ('droplet','letter','cream','toothpaste'):
        mu_model=1e7*e/(2*(1+nu))
        lam_model=1e7*e*nu/((1+nu)*max(1-2*nu,1e-8))
        physical['mu']=mu_model/scale**2
        # Match compute_mu_lam_from_E_nu, including its denominator clamp.
        # The former 1e-4 floor understated bulk modulus above nu=0.49995.
        physical['kappa']=(lam_model+2*mu_model/3)/scale**2
        if scene in ('cream','toothpaste'):
            physical['yield_stress']=1e5*float(params['yield_stress'])/scale**2
    elif scene in ('bird','torus','cat','playdoh'):
        physical={'E':1e7*e/scale**2,'nu':nu}
        if scene in ('cat','playdoh'):
            physical['yield_stress']=1e5*float(params['yield_stress'])/scale**2
    elif scene=='trophy':
        physical={'friction_angle_degrees':float(params['friction_angle'])}
    errors={key:abs(value-gt[key])/abs(gt[key]) for key,value in physical.items()}
    missing=sorted(set(gt)-set(errors))
    return dict(physical=physical,per_parameter_RE=errors,missing_parameters=missing,
        mean_RE=sum(errors.values())/len(errors) if not missing else None,
        partial_mean_RE=sum(errors.values())/len(errors),
        formula='abs(estimated physical parameter - GT) / abs(GT)',
        conversion='E uses 1e7/scale_origin^2; yield uses 1e5/scale_origin^2; nu and friction angle unchanged',
        caveat='Cream/Toothpaste viscosity conversion is not established; full mean RE is intentionally blank' if missing else None)
