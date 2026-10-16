import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
root=Path('results/m6/mesh_cycle_repair')
rows=list(csv.DictReader(open('results/m6/frequency_accuracy_precision/time_half/monitor.csv')))
t=np.array([float(r['time']) for r in rows]);lift=np.array([float(r['lift']) for r in rows])
stats=json.loads((root/'time_analysis/summary.json').read_text())['periodic']['statistics'];b=stats['tip_y']['cycle_boundaries']
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
mask=(t>=b[0])&(t<=b[-1]);ax[0].plot(t[mask],lift[mask]);p=find_peaks(lift,prominence=5)[0];p=p[(t[p]>=b[0])&(t[p]<=b[-1])];ax[0].scatter(t[p],lift[p],s=20,label='Local peaks')
for x in b:ax[0].axvline(x,color='gray',ls='--',lw=.8)
ax[0].set(xlabel='Time (s)',ylabel='Lift (N/m)',title='Multiple peaks per structural cycle');ax[0].legend()
for i,(a,z) in enumerate(zip(b[:-1],b[1:])):
 phase=np.linspace(0,1,401);ax[1].plot(phase,np.interp(a+phase*(z-a),t,lift),label=f'Cycle {i+1}')
ax[1].set(xlabel='Structural phase',ylabel='Lift (N/m)',title=f"Aligned waveform range: {stats['lift']['waveform_variation']:.2%}");ax[1].legend()
fig.savefig(root/'lift_cycles.png',dpi=160)
