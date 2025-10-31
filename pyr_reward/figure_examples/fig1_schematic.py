#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.backends.backend_pdf, matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams["xtick.major.size"] = 10
mpl.rcParams["ytick.major.size"] = 10
plt.rc('font', size=16)          # controls default text sizes
plt.rcParams["font.family"] = "Arial"
fig, axs = plt.subplots(1, 2, figsize=(8,3.5),sharey=True,sharex=True)

# -------------------------
# Place cell schematic
# -------------------------
coms = [40,45,42,52]
reward_locs=[67,100,150,200]
x = np.linspace(0, 270, 270)
y1 = np.exp(-0.5*((x-coms[0])/15)**2) * 2
y2 = np.exp(-0.5*((x-coms[1])/15)**2) * 2
y3 = np.exp(-0.5*((x-coms[2])/15)**2) * 2
y4 = np.exp(-0.5*((x-coms[3])/15)**2) * 2
colors=['k','slategray','darkcyan','darkgoldenrod']

axs[0].plot(x, y1, 'k', label='Epoch 1')
axs[0].plot(x, y2, 'slategray', label='Epoch 2')
axs[0].plot(x, y3, 'darkcyan', label='Epoch 3')
axs[0].plot(x, y4, 'darkgoldenrod', label='Epoch 4')

# mark COMs
for c, col in zip(coms, ['k','slategray','darkcyan']):
    axs[0].axvline(c, color=col,  alpha=0.3)

axs[0].set_title("Place cell selection")
axs[0].set_xlabel("Track position (cm)")
axs[0].set_ylabel("ΔF/F")
# axs[0].legend(frameon=False)
axs[0].set_xlim(0,270)
axs[0].set_xticks([0,270])
axs[0].set_ylim(0,2.5)
axs[0].annotate("Stable position\nacross epochs", 
                xy=(62,1), xytext=(100,1.5),
                arrowprops=dict(arrowstyle="->"))
axs[0].spines[['top','right']].set_visible(False)
for kk,rloc in enumerate(reward_locs):
   axs[0].axvline(rloc, color=colors[kk], ls='--', alpha=0.7)

# -------------------------
# Reward cell schematic
# -------------------------

# COMs are ~10cm before reward
comdist = [35,37,45,32]
coms_reward = [r+comdist[kk] for kk,r in enumerate(reward_locs)]

for i, (rloc, col) in enumerate(zip(reward_locs, colors)):
    y = np.exp(-0.5*((x-(coms_reward[i]))/20)**2) * (2 - 0.2*i)
    axs[1].plot(x, y, col)
    axs[1].axvline(rloc, color=col, ls='--', alpha=0.7, label=f'Reward loc., epoch {i+1}')
    axs[1].axvline(coms_reward[i], color=col, alpha=0.2, label=f'COM, epoch {i+1}')

    # arrow from COM to reward
    axs[1].annotate("",
        xy=(rloc, 0.3+0.2*i),     # reward
        xytext=(coms_reward[i], 0.3+0.2*i), # COM
        arrowprops=dict(arrowstyle="<->", color=col, lw=2))
    
    # label distance
    dist = abs(rloc - coms_reward[i])
    axs[1].text((rloc+coms_reward[i])/2, 0.35+0.2*i,
                fr"$\Delta${dist} cm", ha='center', va='bottom', color=col)

axs[1].set_title("Reward cell selection")
axs[1].set_xlabel("Track position (cm)")
axs[1].set_ylim(0,2.5)
axs[1].set_xlim(0,270)
axs[1].set_xticks([0,270])
# axs[1].legend(frameon=False)
axs[1].spines[['top','right']].set_visible(False)

axs[1].annotate("Stable distance to reward\nacross epochs", 
                xy=(126,1.4), xytext=(110,2.1),
                arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
savedst = r'C:\Users\Han\Box\neuro_phd_stuff\han_2023-\pyramidal_cell_paper\panels_main_figures'
fig.suptitle('Cell classification criteria')
# axs[1].legend()
plt.savefig(os.path.join(savedst, 'place_reward_selection_schematic.svg'))

# %%
    axs[1].axvline(rloc, color=col, ls='--', alpha=0.7)
