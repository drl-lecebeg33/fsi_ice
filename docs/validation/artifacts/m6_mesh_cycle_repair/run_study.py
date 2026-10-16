"""Frozen-source study supervisor; each solver has a separate analysis directory."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from fsi_ice.validation.fsi2_checkpoint import source_digest
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--max-steps',type=int);p.add_argument('--names',nargs='+',default=['baseline','time_half','space_half']);p.add_argument('--validation',required=True);a=p.parse_args()
a.root.mkdir(parents=True,exist_ok=False)
expected=source_digest();env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for f in Path('src/fsi_ice').rglob('*.py'):
 dest=a.root/'frozen_source'/f.relative_to('src');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(folder,meta):(folder/'experiment.json').write_text(json.dumps(meta,indent=2)+'\n')
pending={};records={}
for name in a.names:
 assert name in ['baseline','time_half','space_half']
 folder=a.root/name;folder.mkdir();config=folder/'config.json';shutil.copy2(Path('cases')/f'fsi2_mesh15_{name}.json',config)
 command=[sys.executable,'-u','-m','fsi_ice','fsi2','--config',str(config),'--output',str(folder/'trajectory')]
 if a.max_steps is not None:command+=['--max-steps',str(a.max_steps)]
 with (folder/'run.log').open('w') as log:child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
 meta=dict(status='RUNNING',started_utc=stamp(),source_digest=expected,config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),solver_pid=child.pid,from_zero=True,threads=1,concurrent_runs=len(a.names)>1,fair_timing_evidence=False,validated_by=a.validation,command=command)
 save(folder,meta);pending[name]=child;records[name]=meta
while pending:
 for name,child in list(pending.items()):
  code=child.poll()
  if code is None:continue
  folder=a.root/name;meta=records[name];meta.update(solver_exit_code=code,finished_utc=stamp())
  try:
   assert source_digest()==expected,'Source changed during study'
   command=[sys.executable,'scripts/fsi2_analyze.py',str(folder/'trajectory'),'--output',str(folder/'analysis')]
   result=subprocess.run(command,capture_output=True,text=True,env=env)
   (folder/'analysis.log').write_text(result.stdout+result.stderr);meta['analysis_exit_code']=result.returncode
   summary=json.loads((folder/'analysis'/'summary.json').read_text())
   for k in ['status','accepted_steps','accepted_time','target_time_complete','accepted_physics_passed','solver_failure']:meta[k]=summary.get(k)
  except Exception as exc:meta.update(status='ANALYSIS_ERROR',error=str(exc))
  save(folder,meta);print(name,meta['status'],flush=True);del pending[name]
 if pending:time.sleep(2)
(a.root/'comparison.json').write_text(json.dumps(dict(scope='Same-source mesh-weight-1.5 screening; not three-level convergence or fair timing',source_digest=expected,generated_utc=stamp(),runs=records),indent=2)+'\n')
print('All study processes exited and analysis recorded',flush=True)
