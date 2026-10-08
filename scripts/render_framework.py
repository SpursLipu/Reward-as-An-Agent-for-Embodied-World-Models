"""Render the selected current inference flow as SVG, PNG and PDF."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(18, 10))
fig.patch.set_facecolor('#ffffff')
ax.set(xlim=(0, 18), ylim=(0, 10))
ax.axis('off')
ax.text(.4, 9.55, 'Reward as an Agent', fontsize=27, weight='bold', color='#183848')
ax.text(.4, 9.08, 'current  /  evidence-grounded inference with WMReward + CoTracker3', fontsize=14, color='#526d7c')
ax.text(.4, 8.65, 'Selected evaluation profile: completion audit ON · partial audit OFF · process gate OFF', fontsize=12, color='#526d7c')

def card(x, y, n, title, lines, color):
    ax.add_patch(FancyBboxPatch((x,y), 3.95, 2.8, boxstyle='round,pad=0.02,rounding_size=0.15', facecolor=color, edgecolor='#cad6db', linewidth=1.2))
    ax.text(x+.22,y+2.4,n,fontsize=11,color='#567586',weight='bold')
    ax.text(x+.22,y+1.98,title,fontsize=16,weight='bold',color='#183848')
    for i,line in enumerate(lines):
        ax.text(x+.22,y+1.51-i*.34,line,fontsize=11.5,color='#385766')

def arrow(a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,color='#648492',linewidth=1.8))

xs=[.4,4.8,9.2,13.6]
card(xs[0],5.25,'01 / INPUT','Video + task', ['Original full-duration video','Decode + visibility inspection','Keep task text separate','Invalid/unobservable input → 0'], '#f1f5f7')
card(xs[1],5.25,'02 / PLANNING','Observe + freeze', ['Task-blind visual observations','Extract task requirements','Audit contract coverage','Freeze action, target, end state'], '#fcf4e6')
card(xs[2],5.25,'03 / ASSESSMENT','Grounded draft', ['Task progress / completion','Physical plausibility','Visual quality','Cite supplied source frames'], '#eef5e9')
card(xs[3],5.25,'04 / VERIFICATION','Inspect evidence', ['Refined original frame sequence','Adaptive image crops','CoTracker3 tracks + overlays','Revise the grounded report'], '#eaf3fa')
card(xs[3],1.9,'05 / PHYSICS REFLECTION','WMReward evidence', ['Run verified physical tool','Recheck report against frames','Use tool output as evidence','Tool score is not task reward'], '#eaf3fa')
card(xs[2],1.9,'06 / SCOPE REVIEW','Audit + repair', ['Check each frozen requirement','Recheck contradictions in images','Up to three repair rounds','Retain unresolved audit issues'], '#f0edf8')
card(xs[1],1.9,'07 / COMPLETION CHECK','Conditional audit', ['For complete / mostly complete','Inspect the verification images','Accept same or lower verdict','Reaudit scope; cap upgrades'], '#f0edf8')
card(xs[0],1.9,'08 / REWARD','Resolve + aggregate', ['Conditional failure/progress check','Fixed task / physics / visual rules','Failed or unverifiable content → 0','Protocol faults → review / error'], '#e9f4ef')
for a,b in zip(xs,xs[1:]): arrow((a+3.97,6.65),(b-.04,6.65))
arrow((15.57,5.22),(15.57,4.74))
for a,b in zip(xs[:0:-1],xs[-2::-1]): arrow((a-.03,3.3),(b+4.0,3.3))
ax.text(.4,1.25,'OUTPUT',fontsize=11,weight='bold',color='#567586')
ax.text(1.65,1.25,'score + training_eligible + review reasons + frame-linked reports + tool provenance',fontsize=13,color='#183848')
ax.text(.4,.65,'Local progress must remain valid independently of unresolved task details; it does not establish completion.',fontsize=11,color='#526d7c')
ax.text(.4,.27,'All branches preserve the frozen task and use supplied visual evidence.',fontsize=11,color='#526d7c')
fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
for ext in ('svg','png','pdf'):
    fig.savefig(ROOT/'assets'/f'reward_as_agent_framework.{ext}',dpi=160,facecolor='white')
plt.close(fig)
# Matplotlib emits trailing spaces in SVG path data; keep generated source clean.
svg = ROOT / 'assets/reward_as_agent_framework.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
