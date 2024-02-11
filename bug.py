#%%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')
zip_file = os.getcwd()+"/data/avazu/data/data.zip"
extract_to = os.getcwd()+"/data/avazu"
import zipfile
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_to)


#%%
rdata = pd.read_csv(filepath_or_buffer= extract_to+"/top_app_category.csv")[['reward', 'app_category', 'count']]
rdata['w_reward'] = rdata['reward']*rdata['count']
rdata = rdata.groupby('app_category').sum()
rdata
actual_ctr = np.array(rdata['w_reward']/rdata['count'])
target_v = np.max(actual_ctr)

probData = np.array(rdata['w_reward']/rdata['count'])
countData = np.array(rdata['count'])
rewardData = np.array(rdata['w_reward'])

max_rounds = 1000000
regret = 0
regretVec = np.zeros(max_rounds)
p = len(probData)
n_item_samples = np.zeros(p, dtype='int')
n_item_rewards = np.zeros(p)
n_rounds = 0
seed_idx = 1
np.random.seed(seed_idx)
#%%
for i in range(0,p):
    n_rounds += 1
    regret += target_v - actual_ctr[i]
    regretVec[i] = regret
    x = np.random.binomial(1, probData[i])
    n_item_samples[i] += 1
    countData[i] -= 1
    if x==1:
        n_item_rewards[i] += 1
        rewardData[i] -= 1
probData = rewardData/countData

#%%
results = []
j = 10
for j in tqdm(range(p, max_rounds)):
    exploration = 2 * np.log(n_rounds + 1) / n_item_samples
    exploration = np.power(exploration, 1/2)
    q = n_item_rewards/n_item_samples + exploration
    i = np.argmax(q)
    if (countData[i] == 0):
        break
    
    n_rounds += 1
    regret += target_v - actual_ctr[i]
    regretVec[j] = regret
    x = np.random.binomial(1, probData[i])
    n_item_samples[i] += 1
    countData[i] -= 1
    if x==1:
        n_item_rewards[i] += 1
        rewardData[i] -= 1
    probData[i] =  rewardData[i]/countData[i]   


# %%
import matplotlib.pyplot as plt
plt.plot(regretVec)
# %%
