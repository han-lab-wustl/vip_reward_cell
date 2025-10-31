"""
plt transition data
"""
#%%
import pickle, os, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt,scipy
import matplotlib.backends.backend_pdf, matplotlib as mpl, sys
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as mpatches
sys.path.append(r'C:\Users\Han\Documents\MATLAB\han-lab') ## custom to your clone
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rc('font', size=20)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"

#%%
# get cell eg
fl = r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\Post2Pre_Shuffle"
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\panels_main_figures'
with open(fl, "rb") as fp: #unpickle
    dct = pickle.load(fp)

post2preshuffle=[v for k,v in dct.items()]
post2pre=[0]*len(post2preshuffle)
fl=r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\Pre2Post"
with open(fl, "rb") as fp: #unpickle
    dct = pickle.load(fp)
pre2post=[v for k,v in dct.items()]
fl=r"C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\from_bo\Pre2Post_Shuffle"
with open(fl, "rb") as fp: #unpickle
    dct = pickle.load(fp)
pre2postshuffle=[v for k,v in dct.items()]

df=pd.DataFrame()
df['transition']= np.concatenate([pre2post,post2pre, pre2postshuffle,post2preshuffle])
df['transition']=df['transition']*100
df['cell_type'] = np.concatenate([['Pre to Post']*len(pre2post),['Post to Pre']*len(post2pre), ['Pre to Post']*len(pre2postshuffle),['Post to Pre']*len(post2preshuffle)])
df['shuffle'] = np.concatenate([['Real']*len(pre2post),['Real']*len(post2pre), ['Shuffle']*len(pre2postshuffle),['Shuffle']*len(post2preshuffle)])
df['animal'] = [k for k,v in dct.items()]*4  # assuming all lists are same length
palette = {'Real':'mediumslateblue', 'Shuffle':'grey'
}

s=12
a=.7

# Plot
fig,ax=plt.subplots(figsize=(3,5))
sns.barplot(data=df, x="cell_type", y="transition",hue='shuffle',fill=False,palette=palette,errorbar='se',legend=False)

# Stripplot
sns.stripplot(
    data=df,
    x="cell_type",
    y="transition",
    hue="shuffle",
    dodge=True,
    size=s,
    alpha=a,
    palette=palette
)
# Draw grey lines per animal across real and shuffle
for k,typ in enumerate(['Pre to Post', 'Post to Pre']):
    real_vals = df[(df['cell_type'] == typ) & (df['shuffle'] == 'Real')].sort_values('animal')
    shuf_vals = df[(df['cell_type'] == typ) & (df['shuffle'] == 'Shuffle')].sort_values('animal')
    for i in range(len(real_vals)):
        ax.plot([k-.2, k+.2],
                 [real_vals.iloc[i]['transition'], shuf_vals.iloc[i]['transition']],
                 color='gray', linewidth=1.5, zorder=0,alpha=.5)

ax.set_ylabel("Transition %")
ax.set_xlabel("")
ax.legend()
sns.despine()
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')
# Pre to Post: Real vs Shuffle
pre_real = df[(df['cell_type'] == 'Pre to Post') & (df['shuffle'] == 'Real')]['transition']
pre_shuffle = df[(df['cell_type'] == 'Pre to Post') & (df['shuffle'] == 'Shuffle')]['transition']
kw_pre = scipy.stats.kruskal(pre_real, pre_shuffle)

# Post to Pre: Real vs Shuffle
post_real = df[(df['cell_type'] == 'Post to Pre') & (df['shuffle'] == 'Real')]['transition']
post_shuffle = df[(df['cell_type'] == 'Post to Pre') & (df['shuffle'] == 'Shuffle')]['transition']
stat, p_two_sided = scipy.stats.kruskal(post_real, post_shuffle)
# Compute direction
mean_real = np.mean(post_real)
mean_shuf = np.mean(post_shuffle)

# One-sided p-value: only divide if effect in predicted direction
if mean_real > mean_shuf:
    p_one_sided = p_two_sided / 2
else:
    p_one_sided = 1 - p_two_sided / 2

print(f"One-sided Kruskal-Wallis p = {p_one_sided:.4f} (Real > Shuffle)")
plt.savefig(os.path.join(savedst, 'pre_to_post.svg'), bbox_inches='tight')
