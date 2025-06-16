
import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use 'QtAgg' if this doesn't work
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

## load
data = loadmat('../Data/datatables.mat', struct_as_record=False, squeeze_me=True)
mat_obj = data['SDTdata'][0]
df = [pd.DataFrame({field: getattr(data['SDTdata'][idx], field)
                    for field in mat_obj._fieldnames})
      for idx in range(len(data['SDTdata']))]

## plot block structure
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 4)  # 2 rows, 4 columns
ax = fig.add_subplot(gs[0:2, 0:2])
ax.plot(df[0]['blockSigProb'], df[0]['blockRewProb'], 'ok', markersize=10)
ax.set_xlim(0, 1), ax.set_ylim(0, 1)
ax.tick_params(labelsize=14)
plt.xlabel('Signal probability, $p_s$', fontsize=21), plt.ylabel('Reward probability, $p_r$', fontsize=21)
ax = fig.add_subplot(gs[0, 2:4])
ax.plot(df[0]['blockSigProb'], linewidth=2, color='k')
ax.set_xlim(0, 240), ax.set_ylim(0, 1)
ax.tick_params(labelsize=14), ax.set_xticks(np.arange(0, len(df[0]['blockSigProb']) + 1, 40))
plt.xlabel('Trial', fontsize=21), plt.ylabel('Signal probability, $p_s$', fontsize=21)
ax = fig.add_subplot(gs[1, 2:4])
ax.plot(df[0]['blockRewProb'], linewidth=2, color='k')
ax.set_xlim(0, 240), ax.set_ylim(0, 1)
ax.tick_params(labelsize=14), ax.set_xticks(np.arange(0, len(df[0]['blockRewProb']) + 1, 40))
plt.xlabel('Trial', fontsize=21), plt.ylabel('Reward probability, $p_r$', fontsize=21)
plt.tight_layout()
plt.show()

## plot probability of responding yes
fig = plt.figure(figsize=(18, 6))
choice = np.array([df[idx].choiceBinary for idx in range(len(df))]).flatten()
# as a function of SNR
SNR = np.array([df[idx].SNR for idx in range(len(df))]).flatten()
choice_vs_SNR__mu, choice_vs_SNR__sem = [], []
for this_SNR in np.unique(SNR):
    choice_vs_SNR__mu.append(choice[SNR == this_SNR].mean())
    choice_vs_SNR__sem.append(choice[SNR == this_SNR].std() / np.sqrt(len(df)))
ax = fig.add_subplot(1, 3, 1)
ax.errorbar(np.unique(SNR), choice_vs_SNR__mu, yerr=choice_vs_SNR__sem, linewidth=2, color='k')
ax.set_xlim(-0.1, 0.85), ax.set_ylim(0, 1)
ax.tick_params(labelsize=14)
plt.xlabel('SNR', fontsize=21), plt.ylabel('Probability of responding Yes', fontsize=21)
# as a function of p_s
signalprob = np.array([df[idx].blockSigProb for idx in range(len(df))]).flatten()
choice_vs_signalprob__mu, choice_vs_signalprob__sem = [], []
for this_signalprob in np.unique(signalprob):
    choice_vs_signalprob__mu.append(choice[signalprob == this_signalprob].mean())
    choice_vs_signalprob__sem.append(choice[signalprob == this_signalprob].std() / np.sqrt(len(df)))
ax = fig.add_subplot(1, 3, 2)
ax.errorbar(np.unique(signalprob), choice_vs_signalprob__mu, yerr=choice_vs_signalprob__sem, linewidth=2, color='k')
ax.set_xlim(0, 1), ax.set_ylim(0, 1)
ax.tick_params(labelsize=14)
plt.xlabel('Signal probability, $p_s$', fontsize=21), plt.ylabel('Probability of responding Yes', fontsize=21)
# as a function of p_r
rewardprob = np.array([df[idx].blockRewProb for idx in range(len(df))]).flatten()
choice_vs_rewardprob__mu, choice_vs_rewardprob__sem = [], []
for this_rewardprob in np.unique(rewardprob):
    choice_vs_rewardprob__mu.append(choice[rewardprob == this_rewardprob].mean())
    choice_vs_rewardprob__sem.append(choice[rewardprob == this_rewardprob].std() / np.sqrt(len(df)))
ax = fig.add_subplot(1, 3, 3)
ax.errorbar(np.unique(rewardprob), choice_vs_rewardprob__mu, yerr=choice_vs_rewardprob__sem, linewidth=2, color='k')
ax.set_xlim(0, 1), ax.set_ylim(0, 1)
ax.tick_params(labelsize=14)
plt.xlabel('Reward probability, $p_r$', fontsize=21), plt.ylabel('Probability of responding Yes', fontsize=21)

plt.tight_layout()
plt.show()
