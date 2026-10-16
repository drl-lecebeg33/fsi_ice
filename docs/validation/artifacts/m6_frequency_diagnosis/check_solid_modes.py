import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from fsi_ice.mesh.turek import fsi1_mesh
from fsi_ice.solid.linear import ElasticMaterial,assemble_elasticity,vector_dofs
from fsi_ice.solid.stvk import assemble_mass
from fsi_ice.validation.fsi2_checkpoint import source_digest
root=Path('results/m6/frequency_accuracy')
material=ElasticMaterial(1400000.,.4,density=10000.)
records=[]
for h in (.02,.01,.005):
 pair=fsi1_mesh(h,root/'modal_mesh'/str(h));mesh=pair.solid
 k,_=assemble_elasticity(mesh,material);m=assemble_mass(mesh,material)
 fixed=vector_dofs(mesh.boundary_nodes('solid-clamp')).ravel();free=np.setdiff1d(np.arange(2*mesh.n_nodes),fixed)
 kk=k[free][:,free];mm=m[free][:,free]
 vals,vec=eigh(kk.toarray(),mm.toarray(),subset_by_index=[0,2]);res=[]
 for v,x in zip(vals,vec.T):
  res.append(float(np.linalg.norm(kk@x-v*(mm@x))/(np.linalg.norm(kk@x)+np.linalg.norm(v*(mm@x)))))
 item={'h':h,'solid_nodes':mesh.n_nodes,'solid_cells':mesh.n_cells,'fluid_nodes':pair.fluid.n_nodes,'fluid_cells':pair.fluid.n_cells,'clamp_nodes':len(fixed)//2,'frequencies_hz':(np.sqrt(vals)/(2*np.pi)).tolist(),'eigen_residuals_relative':res}
 records.append(item);print(json.dumps(item),flush=True)
result={'scope':'Dry, undeformed, small-strain solid eigenmodes: diagnostic only, not FSI2 frequencies or physical acceptance','source_digest':source_digest(),'material':{'young':1400000.,'poisson':.4,'density':10000.,'model':'plane_strain'},'records':records}
(root/'solid_modes.json').write_text(json.dumps(result,indent=2)+'\n')
