import json,numpy as np
from pathlib import Path
from scipy.sparse.linalg import spsolve
from fsi_ice.mesh.turek import read_fsi_gmsh2
from fsi_ice.solid.linear import ElasticMaterial,assemble_elasticity,vector_dofs
root=Path('results/m6/frequency_accuracy');records=[]
for h in (.02,.01,.005):
 mesh=read_fsi_gmsh2(root/'modal_mesh'/str(h)/'fsi1.msh').solid
 k,_=assemble_elasticity(mesh,ElasticMaterial(1400000.,.4,density=10000.))
 free=np.setdiff1d(np.arange(2*mesh.n_nodes),vector_dofs(mesh.boundary_nodes('solid-clamp')).ravel())
 rhs=np.zeros(2*mesh.n_nodes)
 for edge in mesh.boundaries['solid-interface']:
  if np.allclose(mesh.points[edge,0],.6,rtol=0,atol=1e-12):
   length=np.linalg.norm(mesh.points[edge[0]]-mesh.points[edge[1]])
   rhs[2*edge+1]+=length*.5/.02
 assert np.isclose(rhs[1::2].sum(),1.,atol=1e-12)
 d=np.zeros_like(rhs);d[free]=spsolve(k[free][:,free],rhs[free]);tip=np.argmin(np.linalg.norm(mesh.points-[.6,.2],axis=1))
 residual=float(np.linalg.norm((k@d-rhs)[free])/np.linalg.norm(rhs[free]));assert residual<1e-7
 item={'h':h,'vertical_force_n':float(rhs[1::2].sum()),'tip_y_compliance_m_per_n':float(d[2*tip+1]),'total_solid_mass_kg':float(mesh.areas.sum()*10000.),'relative_free_residual':residual};records.append(item);print(json.dumps(item))
(root/'solid_compliance.json').write_text(json.dumps({'scope':'Diagnostic dry linear beam only; not a standard FSI2 solution or acceptance test','records':records},indent=2)+'\n')
