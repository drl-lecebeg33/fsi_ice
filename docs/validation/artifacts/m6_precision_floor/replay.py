import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from fsi_ice.validation.fsi2 import build_fsi2_problem
from fsi_ice.validation.fsi2_checkpoint import load_checkpoint,source_digest
from fsi_ice.coupling.newton import NewtonOptions
import fsi_ice.coupling.transient as tr
parser=argparse.ArgumentParser();parser.add_argument('name');parser.add_argument('--inner',type=float);parser.add_argument('--share',type=float);parser.add_argument('--label',default='');parser.add_argument('--base');parser.add_argument('--steps',type=int,default=1);args=parser.parse_args()
root=Path('results/m6/precision_floor')/(args.name+args.label+('_share'+str(args.share) if args.share else ('_inner'+str(args.inner) if args.inner else '_original')));root.mkdir(exist_ok=False)
base=Path(args.base) if args.base else Path('results/m6/frequency_accuracy_repaired')/args.name
with np.load(base/'trajectory/checkpoint.npz',allow_pickle=False) as data: stored_identity=json.loads(str(data['metadata'].item()))
cfg=json.loads((base/'config.json').read_text());driver=build_fsi2_problem(cfg,root)
mesh_hash=hashlib.sha256((root/'mesh/fsi1.msh').read_bytes()).hexdigest();state=load_checkpoint(base/'trajectory/checkpoint.npz',config=cfg,mesh_sha256=mesh_hash,code_sha256=stored_identity['code_sha256']);driver.coordinator.load(state)
original=tr.solve_dirichlet_neumann;records=[]
original_solid=tr.solve_solid_step
solid_calls=[]
def capture_solid(*a,**kw):
 try:
  out=original_solid(*a,**kw)
  solid_calls.append({'absolute':kw.get('absolute_tolerance'),'relative':out[1].nonlinear_history[-1]})
  return out
 except Exception as exc:
  tb=exc.__traceback__
  while tb and tb.tb_frame.f_code.co_name!='solve_stvk':tb=tb.tb_next
  if tb:
   v=tb.tb_frame.f_locals
   from scipy.sparse import save_npz
   free=v['free'];r=v['residual'];delta=v.get('correction',np.zeros_like(v['values']))
   info={'absolute_target':v['absolute_tolerance'],'relative_target':v['tolerance'],'scale':float(v['scale']),'raw_residual':float(np.linalg.norm(r[free])),'max_correction':float(np.max(np.abs(delta))),'max_displacement':float(np.max(np.abs(v['values']))),'nonlinear_history':v['history'],'prior_solid_calls':solid_calls}
   (root/'solid_failure.json').write_text(json.dumps(info,indent=2)+'\n')
   np.savez(root/'solid_failure.npz',points=a[0].points,cells=a[0].cells,external=v['external'],fixed_nodes=v['fixed_nodes'],initial=kw['initial'],values=v['values'],residual=r,correction=delta,mass_rhs=v['mass_rhs'],mass_coefficient=v['mass_coefficient'])
   save_npz(root/'tangent.npz',v['matrix'])
  raise
tr.solve_solid_step=capture_solid
def instrument(evaluate,initial,**kw):
 check=kw['convergence_check'];context=[c.cell_contents for c in check.__closure__ if type(c.cell_contents).__name__=='MonolithicContext'][0]
 def observed(x):
  result=evaluate(x);res=tr.evaluate_monolithic(result.fluid,context=context,mode='residual').residual
  gap=result.target-x;scales=kw['scales'];mag=max(np.max(np.abs(x/scales)),np.max(np.abs(result.target/scales)));opt=kw['options'];threshold=opt.absolute_tolerance+opt.relative_tolerance*mag
  blocks={name:float(np.max(np.abs(res[driver.layout.slice_of(name)]/driver.scales[driver.layout.slice_of(name)]))) for name in ('u_f','p_f','d_s','v_s','d_m','g')}
  r={'step':driver.coordinator.state.step+1,'evaluation_index':len(records)+1,'gap':float(np.max(np.abs(gap/scales))),'threshold':float(threshold),'full_norm':max(blocks.values()),'blocks':blocks};records.append(r)
  if len(records)<20 or len(records)%10==0:print(json.dumps(r),flush=True)
  (root/'iterations.json').write_text(json.dumps(records,indent=2)+'\n');return result
 return original(observed,initial,**kw)
tr.solve_dirichlet_neumann=instrument
if args.inner:cfg['solver']['single_field_tolerance']=args.inner
if args.share:tr._INNER_GATE_SHARE=args.share
s=cfg['solver'];a=cfg['acceptance'];beg=time.perf_counter()
try:
 for index in range(args.steps):
  out=driver.step(cfg['method'],coupling=cfg['coupling'],newton=NewtonOptions(atol=s['newton_atol'],rtol=s['newton_rtol'],max_iterations=s['max_iterations']),single_tolerance=s['single_field_tolerance'],residual_max=a['residual_max'],continuity_max=a['continuity_max'],minimum_quality=a['minimum_mesh_quality'])
 result={'status':'ACCEPTED_DIAGNOSTIC_ONLY','step':out.step,'monitor':out.monitoring[-1],'new_monitoring':list(out.monitoring[state.step:])}
except Exception as exc:
 result={'status':'FAIL','error':str(exc),'rollback_step':driver.coordinator.state.step}
result.update(checkpoint_source_digest=stored_identity['code_sha256'],checkpoint_sha256=hashlib.sha256((base/'trajectory/checkpoint.npz').read_bytes()).hexdigest(),source_digest=source_digest(),seconds=time.perf_counter()-beg,original_step=state.step,config=cfg,scope='Isolated candidate replay, not a resumable new-source trajectory')
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
