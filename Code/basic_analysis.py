
import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Use 'QtAgg' if this doesn't work
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

## load data
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
choice_vs_signalprob_signaltrials__mu, choice_vs_signalprob_signaltrials__sem = [], []
choice_vs_signalprob_noisetrials__mu, choice_vs_signalprob_noisetrials__sem = [], []
for this_signalprob in np.unique(signalprob):
    choice_vs_signalprob_signaltrials__mu.append(choice[np.logical_and(signalprob == this_signalprob, SNR != 0)].mean())
    choice_vs_signalprob_signaltrials__sem.append(choice[np.logical_and(signalprob == this_signalprob, SNR != 0)].std() / np.sqrt(len(df)))
    choice_vs_signalprob_noisetrials__mu.append(choice[np.logical_and(signalprob == this_signalprob, SNR == 0)].mean())
    choice_vs_signalprob_noisetrials__sem.append(choice[np.logical_and(signalprob == this_signalprob, SNR == 0)].std() / np.sqrt(len(df)))
ax = fig.add_subplot(1, 3, 2)
ax.errorbar(np.unique(signalprob), choice_vs_signalprob_signaltrials__mu, yerr=choice_vs_signalprob_signaltrials__sem, linewidth=2, color='k', label='signal trials (SNR > 0)')
ax.errorbar(np.unique(signalprob), choice_vs_signalprob_noisetrials__mu, yerr=choice_vs_signalprob_noisetrials__sem, linewidth=2, color='k', linestyle='--', label='noise trials (SNR = 0)')
ax.set_xlim(0, 1), ax.set_ylim(0, 1)
ax.tick_params(labelsize=14)
plt.legend(fontsize=18)
plt.xlabel('Signal probability, $p_s$', fontsize=21), plt.ylabel('Probability of responding Yes', fontsize=21)
# as a function of p_r
rewardprob = np.array([df[idx].blockRewProb for idx in range(len(df))]).flatten()
choice_vs_rewardprob_signaltrials__mu, choice_vs_rewardprob_signaltrials__sem = [], []
choice_vs_rewardprob_noisetrials__mu, choice_vs_rewardprob_noisetrials__sem = [], []
for this_rewardprob in np.unique(rewardprob):
    choice_vs_rewardprob_signaltrials__mu.append(choice[np.logical_and(rewardprob == this_rewardprob, SNR != 0)].mean())
    choice_vs_rewardprob_signaltrials__sem.append(choice[np.logical_and(rewardprob == this_rewardprob, SNR != 0)].std() / np.sqrt(len(df)))
    choice_vs_rewardprob_noisetrials__mu.append(choice[np.logical_and(rewardprob == this_rewardprob, SNR == 0)].mean())
    choice_vs_rewardprob_noisetrials__sem.append(choice[np.logical_and(rewardprob == this_rewardprob, SNR == 0)].std() / np.sqrt(len(df)))
ax = fig.add_subplot(1, 3, 3)
ax.errorbar(np.unique(rewardprob), choice_vs_rewardprob_signaltrials__mu, yerr=choice_vs_rewardprob_signaltrials__sem, linewidth=2, color='k', label='signal trials (SNR > 0)')
ax.errorbar(np.unique(rewardprob), choice_vs_rewardprob_noisetrials__mu, yerr=choice_vs_rewardprob_noisetrials__sem, linewidth=2, color='k', linestyle='--', label='noise trials (SNR = 0)')
ax.set_xlim(0, 1), ax.set_ylim(0, 1)
ax.tick_params(labelsize=14)
plt.legend(fontsize=18)
plt.xlabel('Reward probability, $p_r$', fontsize=21), plt.ylabel('Probability of responding Yes', fontsize=21)
plt.tight_layout()
plt.show()

## plot perceptual confidence
fig = plt.figure(figsize=(18, 6))
perceptualConf = np.array([df[idx].perceptualConf for idx in range(len(df))]).flatten()
# as a function of SNR
SNR = np.array([df[idx].SNR for idx in range(len(df))]).flatten()
perceptualConf_vs_SNR__mu, perceptualConf_vs_SNR__sem = [], []
for this_SNR in np.unique(SNR):
    perceptualConf_vs_SNR__mu.append(perceptualConf[SNR == this_SNR].mean())
    perceptualConf_vs_SNR__sem.append(perceptualConf[SNR == this_SNR].std() / np.sqrt(len(df)))
ax = fig.add_subplot(1, 3, 1)
ax.errorbar(np.unique(SNR), perceptualConf_vs_SNR__mu, yerr=perceptualConf_vs_SNR__sem, linewidth=2, color='k')
ax.set_xlim(-0.1, 0.85), ax.set_ylim(0, 5)
ax.tick_params(labelsize=14)
plt.xlabel('SNR', fontsize=21), plt.ylabel('Perceptual confidence', fontsize=21)
# as a function of p_s
signalprob = np.array([df[idx].blockSigProb for idx in range(len(df))]).flatten()
perceptualConf_vs_signalprob_signaltrials__mu, perceptualConf_vs_signalprob_signaltrials__sem = [], []
perceptualConf_vs_signalprob_noisetrials__mu, perceptualConf_vs_signalprob_noisetrials__sem = [], []
for this_signalprob in np.unique(signalprob):
    perceptualConf_vs_signalprob_signaltrials__mu.append(perceptualConf[np.logical_and(signalprob == this_signalprob, SNR != 0)].mean())
    perceptualConf_vs_signalprob_signaltrials__sem.append(perceptualConf[np.logical_and(signalprob == this_signalprob, SNR != 0)].std() / np.sqrt(len(df)))
    perceptualConf_vs_signalprob_noisetrials__mu.append(perceptualConf[np.logical_and(signalprob == this_signalprob, SNR == 0)].mean())
    perceptualConf_vs_signalprob_noisetrials__sem.append(perceptualConf[np.logical_and(signalprob == this_signalprob, SNR == 0)].std() / np.sqrt(len(df)))
ax = fig.add_subplot(1, 3, 2)
ax.errorbar(np.unique(signalprob), perceptualConf_vs_signalprob_signaltrials__mu, yerr=perceptualConf_vs_signalprob_signaltrials__sem, linewidth=2, color='k', label='signal trials (SNR > 0)')
ax.errorbar(np.unique(signalprob), perceptualConf_vs_signalprob_noisetrials__mu, yerr=perceptualConf_vs_signalprob_noisetrials__sem, linewidth=2, color='k', linestyle='--', label='noise trials (SNR = 0)')
ax.set_xlim(0, 1), ax.set_ylim(0, 5)
ax.tick_params(labelsize=14)
plt.legend(fontsize=18)
plt.xlabel('Signal probability, $p_s$', fontsize=21), plt.ylabel('Perceptual confidence', fontsize=21)
# as a function of p_r
rewardprob = np.array([df[idx].blockRewProb for idx in range(len(df))]).flatten()
perceptualConf_vs_rewardprob_signaltrials__mu, perceptualConf_vs_rewardprob_signaltrials__sem = [], []
perceptualConf_vs_rewardprob_noisetrials__mu, perceptualConf_vs_rewardprob_noisetrials__sem = [], []
for this_rewardprob in np.unique(rewardprob):
    perceptualConf_vs_rewardprob_signaltrials__mu.append(perceptualConf[np.logical_and(rewardprob == this_rewardprob, SNR != 0)].mean())
    perceptualConf_vs_rewardprob_signaltrials__sem.append(perceptualConf[np.logical_and(rewardprob == this_rewardprob, SNR != 0)].std() / np.sqrt(len(df)))
    perceptualConf_vs_rewardprob_noisetrials__mu.append(perceptualConf[np.logical_and(rewardprob == this_rewardprob, SNR == 0)].mean())
    perceptualConf_vs_rewardprob_noisetrials__sem.append(perceptualConf[np.logical_and(rewardprob == this_rewardprob, SNR == 0)].std() / np.sqrt(len(df)))
ax = fig.add_subplot(1, 3, 3)
ax.errorbar(np.unique(rewardprob), perceptualConf_vs_rewardprob_signaltrials__mu, yerr=perceptualConf_vs_rewardprob_signaltrials__sem, linewidth=2, color='k', label='signal trials (SNR > 0)')
ax.errorbar(np.unique(rewardprob), perceptualConf_vs_rewardprob_noisetrials__mu, yerr=perceptualConf_vs_rewardprob_noisetrials__sem, linewidth=2, color='k', linestyle='--', label='noise trials (SNR = 0)')
ax.set_xlim(0, 1), ax.set_ylim(0, 5)
ax.tick_params(labelsize=14)
plt.legend(fontsize=18)
plt.xlabel('Reward probability, $p_r$', fontsize=21), plt.ylabel('Perceptual confidence', fontsize=21)
plt.tight_layout()
plt.show()

## plot reward confidence
fig = plt.figure(figsize=(18, 6))
rewardConf = np.array([df[idx].rewardConf for idx in range(len(df))]).flatten()
# as a function of SNR
SNR = np.array([df[idx].SNR for idx in range(len(df))]).flatten()
rewardConf_vs_SNR__mu, rewardConf_vs_SNR__sem = [], []
for this_SNR in np.unique(SNR):
    rewardConf_vs_SNR__mu.append(rewardConf[SNR == this_SNR].mean())
    rewardConf_vs_SNR__sem.append(rewardConf[SNR == this_SNR].std() / np.sqrt(len(df)))
ax = fig.add_subplot(1, 3, 1)
ax.errorbar(np.unique(SNR), rewardConf_vs_SNR__mu, yerr=rewardConf_vs_SNR__sem, linewidth=2, color='k')
ax.set_xlim(-0.1, 0.85), ax.set_ylim(0, 5)
ax.tick_params(labelsize=14)
plt.xlabel('SNR', fontsize=21), plt.ylabel('Reward confidence', fontsize=21)
# as a function of p_s
signalprob = np.array([df[idx].blockSigProb for idx in range(len(df))]).flatten()
rewardConf_vs_signalprob_signaltrials__mu, rewardConf_vs_signalprob_signaltrials__sem = [], []
rewardConf_vs_signalprob_noisetrials__mu, rewardConf_vs_signalprob_noisetrials__sem = [], []
for this_signalprob in np.unique(signalprob):
    rewardConf_vs_signalprob_signaltrials__mu.append(rewardConf[np.logical_and(signalprob == this_signalprob, SNR != 0)].mean())
    rewardConf_vs_signalprob_signaltrials__sem.append(rewardConf[np.logical_and(signalprob == this_signalprob, SNR != 0)].std() / np.sqrt(len(df)))
    rewardConf_vs_signalprob_noisetrials__mu.append(rewardConf[np.logical_and(signalprob == this_signalprob, SNR == 0)].mean())
    rewardConf_vs_signalprob_noisetrials__sem.append(rewardConf[np.logical_and(signalprob == this_signalprob, SNR == 0)].std() / np.sqrt(len(df)))
ax = fig.add_subplot(1, 3, 2)
ax.errorbar(np.unique(signalprob), rewardConf_vs_signalprob_signaltrials__mu, yerr=rewardConf_vs_signalprob_signaltrials__sem, linewidth=2, color='k', label='signal trials (SNR > 0)')
ax.errorbar(np.unique(signalprob), rewardConf_vs_signalprob_noisetrials__mu, yerr=rewardConf_vs_signalprob_noisetrials__sem, linewidth=2, color='k', linestyle='--', label='noise trials (SNR = 0)')
ax.set_xlim(0, 1), ax.set_ylim(0, 5)
ax.tick_params(labelsize=14)
plt.legend(fontsize=18)
plt.xlabel('Signal probability, $p_s$', fontsize=21), plt.ylabel('Reward confidence', fontsize=21)
# as a function of p_r
rewardprob = np.array([df[idx].blockRewProb for idx in range(len(df))]).flatten()
rewardConf_vs_rewardprob_signaltrials__mu, rewardConf_vs_rewardprob_signaltrials__sem = [], []
rewardConf_vs_rewardprob_noisetrials__mu, rewardConf_vs_rewardprob_noisetrials__sem = [], []
for this_rewardprob in np.unique(rewardprob):
    rewardConf_vs_rewardprob_signaltrials__mu.append(rewardConf[np.logical_and(rewardprob == this_rewardprob, SNR != 0)].mean())
    rewardConf_vs_rewardprob_signaltrials__sem.append(rewardConf[np.logical_and(rewardprob == this_rewardprob, SNR != 0)].std() / np.sqrt(len(df)))
    rewardConf_vs_rewardprob_noisetrials__mu.append(rewardConf[np.logical_and(rewardprob == this_rewardprob, SNR == 0)].mean())
    rewardConf_vs_rewardprob_noisetrials__sem.append(rewardConf[np.logical_and(rewardprob == this_rewardprob, SNR == 0)].std() / np.sqrt(len(df)))
ax = fig.add_subplot(1, 3, 3)
ax.errorbar(np.unique(rewardprob), rewardConf_vs_rewardprob_signaltrials__mu, yerr=rewardConf_vs_rewardprob_signaltrials__sem, linewidth=2, color='k', label='signal trials (SNR > 0)')
ax.errorbar(np.unique(rewardprob), rewardConf_vs_rewardprob_noisetrials__mu, yerr=rewardConf_vs_rewardprob_noisetrials__sem, linewidth=2, color='k', linestyle='--', label='noise trials (SNR = 0)')
ax.set_xlim(0, 1), ax.set_ylim(0, 5)
ax.tick_params(labelsize=14)
plt.legend(fontsize=18)
plt.xlabel('Reward probability, $p_r$', fontsize=21), plt.ylabel('Reward confidence', fontsize=21)
plt.tight_layout()
plt.show()

## plot evolution of choice, perceptual confidence, and reward confidence
fig = plt.figure(figsize=(12, 36))
blockSigProb = np.array([df[idx].blockSigProb for idx in range(len(df))]).mean(axis=0)
blockRewProb = np.array([df[idx].blockRewProb for idx in range(len(df))]).mean(axis=0)
choice_mu = np.array([df[idx].choiceBinary for idx in range(len(df))]).mean(axis=0)
choice_sem = np.array([df[idx].choiceBinary for idx in range(len(df))]).std(axis=0) / np.sqrt(len(df))
perceptualConf_mu = np.array([df[idx].perceptualConf for idx in range(len(df))]).mean(axis=0)
perceptualConf_sem = np.array([df[idx].perceptualConf for idx in range(len(df))]).std(axis=0) / np.sqrt(len(df))
rewardConf_mu = np.array([df[idx].rewardConf for idx in range(len(df))]).mean(axis=0)
rewardConf_sem = np.array([df[idx].rewardConf for idx in range(len(df))]).std(axis=0) / np.sqrt(len(df))
ax = fig.add_subplot(5, 1, 1)
ax.plot(range(len(blockSigProb)), blockSigProb, linewidth=2, color='k')
ax = fig.add_subplot(5, 1, 2)
ax.plot(range(len(blockRewProb)), blockRewProb, linewidth=2, color='k')
ax = fig.add_subplot(5, 1, 3)
ax.errorbar(range(len(choice_mu)), choice_mu, yerr=choice_sem, linewidth=2, color='k')
ax.set_ylim(0, 1)
ax = fig.add_subplot(5, 1, 4)
ax.errorbar(range(len(perceptualConf_mu)), perceptualConf_mu, yerr=perceptualConf_sem, linewidth=2, color='k')
ax.set_ylim(2, 4)
ax = fig.add_subplot(5, 1, 5)
ax.errorbar(range(len(rewardConf_mu)), rewardConf_mu, yerr=rewardConf_sem, linewidth=2, color='k')
ax.set_ylim(2, 4)