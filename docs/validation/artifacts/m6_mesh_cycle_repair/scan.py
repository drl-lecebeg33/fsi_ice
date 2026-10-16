import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from fsi_ice.validation.fsi2 import build_fsi2_problem
from fsi_ice.validation.fsi2_checkpoint import load_checkpoint,source_digest
from fsi_ice.coupling.newton import NewtonOptions
import fsi_ice.coupling.transient as tr
parser=argparse.ArgumentParser();parser.add_argument('name');parser.add_argument('--inner',type=float);parser.add_argument('--share',type=float);parser.add_argument('--label',default='');parser.add_argument('--base');parser.add_argument('--diagnostic-low-volume',action='store_true');parser.add_argument('--relax',type=float);parser.add_argument('--steps',type=int,default=1);args=parser.parse_args()
root=Path('results/m6/mesh_cycle_repair')/(args.name+args.label+('_share'+str(args.share) if args.share else ('_inner'+str(args.inner) if args.inner else '_original')));root.mkdir(exist_ok=False)
base=Path(args.base) if args.base else Path('results/m6/frequency_accuracy_repaired')/args.name
with np.load(base/'checkpoint.npz',allow_pickle=False) as data: stored_identity=json.loads(str(data['metadata'].item()))
cfg=json.loads((base/'config.json').read_text());driver=build_fsi2_problem(cfg,root)
mesh_hash=hashlib.sha256((root/'mesh/fsi1.msh').read_bytes()).hexdigest();state=load_checkpoint(base/'checkpoint.npz',config=cfg,mesh_sha256=mesh_hash,code_sha256=stored_identity['code_sha256']);driver.coordinator.load(state)

from fsi_ice.ale.harmonic import build_interface_weighted_extension
from fsi_ice.ale.geometry import build_ale_geometry
fluid=driver.topology.fluid_mesh;nodes=fluid.boundary_nodes();indices={int(n):i for i,n in enumerate(nodes)}
boundary=np.zeros((len(nodes),2))
for sn,fn in zip(driver.topology.solid_nodes,driver.topology.fluid_vertex_nodes): boundary[indices[int(fn)]]=state.solid_state[0][sn]
results=[]
for length,power in [(.005,1),(.005,1.5),(.005,2),(.01,2),(.02,2),(.01,3)]:
 extension=build_interface_weighted_extension(fluid,interface_tag='fluid-interface',length=length,power=power)
 for scale in [1.,1.15,1.3]:
  dm=extension.extend(nodes,boundary*scale)
  try: geometry=build_ale_geometry(fluid,dm.values,quality_limits=dict(min_volume_ratio=1e-6,min_quality=1e-6),candidate_id='geometry-only')
  except ValueError as exc:
   results.append(dict(length=length,power=power,displacement_scale=scale,error=str(exc)));continue
  row=dict(length=length,power=power,displacement_scale=scale,min_volume=geometry.minimum_volume_ratio,min_quality=geometry.minimum_quality,residual=max(dm.component_residuals));results.append(row);print(row,flush=True)
(root/'geometry_scan.json').write_text(json.dumps(results,indent=2)+'\n')
