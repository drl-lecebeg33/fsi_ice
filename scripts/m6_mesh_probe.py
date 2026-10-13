"""Read-only mesh-motion comparison using saved FSI2 beam motion.

Scale factors are artificial boundary inputs; no extra FSI states are claimed.
The original checkpoint and its failure record are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fsi_ice.ale import build_harmonic_extension, build_interface_weighted_extension
from fsi_ice.fem.p2 import P2Space
from fsi_ice.interface import build_matching_interface
from fsi_ice.mesh.turek import read_fsi_gmsh2


def inspect(folder, output, *, length=.005, power=1.):
    mesh_file=folder/'mesh/fsi1.msh'
    checkpoint_file=folder/'checkpoint.npz'
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (mesh_file,checkpoint_file,folder/'failure.json') if p.exists()}
    pair=read_fsi_gmsh2(mesh_file)
    topology=build_matching_interface(pair.solid,pair.fluid,P2Space.from_mesh(pair.fluid),
                                      solid_tag='solid-interface',fluid_tag='fluid-interface',atol=1e-12,rtol=0.)
    nodes=pair.fluid.boundary_nodes()
    positions={int(node):i for i,node in enumerate(nodes)}
    rows=[]
    with np.load(checkpoint_file,allow_pickle=False) as data:
        histories=np.array(data['solid_history'],copy=True)
        metadata=json.loads(str(data['metadata'].item()))
    if hashlib.sha256(mesh_file.read_bytes()).hexdigest()!=metadata['mesh_sha256']:
        raise ValueError('Mesh differs from accepted checkpoint.')
    extensions={'harmonic':build_harmonic_extension(pair.fluid),
                'weighted_harmonic':build_interface_weighted_extension(pair.fluid,
                    interface_tag='fluid-interface',length=length,power=power)}
    for layer in reversed(range(len(histories))):
        displacement=histories[layer,0]
        boundary=np.zeros((len(nodes),2))
        for solid_node,fluid_node in zip(topology.solid_nodes,topology.fluid_vertex_nodes):
            boundary[positions[int(fluid_node)]]=displacement[solid_node]
        for name,extension in extensions.items():
            mesh_displacement=extension.extend(nodes,boundary).values
            for scale in (1.,1.25,1.5,2.,3.,-1.,-2.):
                vertices=(pair.fluid.points+scale*mesh_displacement)[pair.fluid.cells]
                first=vertices[:,1]-vertices[:,0]
                second=vertices[:,2]-vertices[:,0]
                determinant=first[:,0]*second[:,1]-first[:,1]*second[:,0]
                ratio=determinant/pair.fluid.determinants
                edge=second-first
                quality=(2*np.sqrt(3)*determinant /
                         (np.sum(first*first,axis=1)+np.sum(second*second,axis=1)+
                          np.sum(edge*edge,axis=1)))
                rows.append({'saved_step':metadata['step']-layer,'method':name,
                             'artificial_scale':scale,'minimum_area_ratio':float(ratio.min()),
                             'minimum_quality':float(quality.min()),
                             'worst_cell':int(np.argmin(ratio)),
                             'passes_existing_gates':bool(np.min(ratio)>=.2 and np.min(quality)>=.1)})
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in (mesh_file,checkpoint_file,folder/'failure.json') if p.exists()}
    if before!=after:
        raise RuntimeError('Input files changed during read-only mesh analysis.')
    result={'source':str(folder),'input_sha256':before,'input_unchanged':True,
            'length':length,'power':power,'scope':'saved boundary displacements and artificial scales only',
            'rows':rows}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--length',type=float,default=.005)
    parser.add_argument('--power',type=float,default=1.)
    args=parser.parse_args()
    result=inspect(args.folder,args.output,length=args.length,power=args.power)
    print(json.dumps({'input_unchanged':result['input_unchanged'],
                      'rows':len(result['rows'])}))
