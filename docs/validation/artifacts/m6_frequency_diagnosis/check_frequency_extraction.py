"""Read-only baseline check; run with project .venv from the repository root."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
rows=list(csv.DictReader(open('results/m6/physical_gate/iqn_12s/trajectory/monitor.csv')))
t=np.array([float(r['time']) for r in rows]);y=np.array([float(r['tip_y']) for r in rows]);select=t>=10.5;tt=t[select];yy=y[select]
def fit(f):
    a=np.column_stack([np.ones_like(tt)]+[g(2*np.pi*j*f*tt) for j in range(1,4) for g in (np.sin,np.cos)])
    c=np.linalg.lstsq(a,yy,rcond=None)[0]
    return np.linalg.norm(a@c-yy)**2
opt=minimize_scalar(fit,bounds=(1.8,2.5),method='bounded',options={'xatol':1e-11})
center=float(yy.mean());idx=np.flatnonzero((yy[:-1]<center)&(yy[1:]>=center));cross=tt[idx]+(center-yy[idx])/(yy[idx+1]-yy[idx])*(tt[idx+1]-tt[idx]);freq=1/np.mean(np.diff(cross))
assert opt.success and len(cross)>=3
assert abs(opt.x-2.1841817795361025)/2.1841817795361025<1e-4
assert abs(freq-2.1841817795361025)/2.1841817795361025<1e-4
print(json.dumps({'fit_frequency_hz':float(opt.x),'crossing_frequency_hz':float(freq),'fit_rms_m':float(np.sqrt(opt.fun/len(tt)))},indent=2))
