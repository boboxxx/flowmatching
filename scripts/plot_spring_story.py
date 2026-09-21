"""Publication plots from frozen paired CSVs; no synthetic points."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'experiments/spring_final_20260921/evidence.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
 'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,
 'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.5})
fig,axes=plt.subplots(1,2,figsize=(7.05,2.05),layout='constrained')
for name,label,color,marker in zip(('div2k','kodak'),('DIV2K','Kodak24'),('#0072B2','#D55E00'),('o','s')):
 r=data[name]
 axes[0].plot([p['snr'] for p in r['per_snr']], [100*(p['adaptive_harq']['nack']-p['flowharq']['nack']) for p in r['per_snr']],color=color,marker=marker,markersize=4,label=label)
 axes[1].plot([p['eta'] for p in r['overhead']], [100*p['relative_saving'] for p in r['overhead']],color=color,marker=marker,markersize=4,label=label)
 axes[1].axhline(100*r['payload_relative_saving'],color=color,ls='--',lw=.9)
axes[0].set(xlabel='Nominal SNR (dB)',ylabel='NACK reduction (pp)',xticks=[0,3,6,9,12,15])
axes[0].axhline(0,color='0.5',lw=.6)
axes[1].set(xlabel='Metadata/control efficiency (bits/complex use)',ylabel='Charged-use saving (%)',xscale='log',xticks=[.25,.5,1,2,4])
axes[1].set_xticklabels(['0.25','0.5','1','2','4'])
axes[1].text(.03,.90,'Dashed: payload only',transform=axes[1].transAxes,va='top',fontsize=7,bbox={'facecolor':'white','edgecolor':'none','pad':1})
for ax in axes:
 ax.grid(axis='y',alpha=.18)
 ax.legend(loc='lower right',frameon=False)
for extension in ('pdf','png'):
 fig.savefig(ROOT/f'paper/figures/spring_audit.{extension}',dpi=300)
print('Saved spring_audit.pdf and .png')
