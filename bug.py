#%%
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')
#from numba import jit
zip_file = os.getcwd()+"/data/avazu/data/data.zip"
extract_to = os.getcwd()+"/data/avazu"
import zipfile
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_to)


#%%
@jit
#def ucb1_df(n_iterations, rounds):
main_df = np.genfromtxt(extract_to+"/top_app_category.csv",
                        delimiter=',',
                        dtype=str,
                        encoding='UTF-8')
main_df = np.delete(main_df, 0, axis=0)
main_df = np.delete(main_df, [0,4], axis=1)
main_df[:,0] = main_df[:,0].astype(int)

item_col_name = 2

items = np.unique(main_df[:,item_col_name])
n_items = len(items)

actual_ctr = []
for app in items:
    idx = np.where(main_df[:,2]==app)[0]
    actual_ctr.append(np.mean(main_df[idx][:,0].astype(int)))

#%%
n_iterations = 10000
results = []
iteration = 0
for iteration in tqdm(range(n_iterations)):
    
    np.random.seed(iteration)
    sample_df = np.random.permutation(main_df)
    rewardVec = sample_df[:,0].astype('int')
    armVec = sample_df[:,2]
    index_list = list()
    index_list_len = list()
#    index_list_counts = np.zeros(n_items)

    for app in items:
        idx = np.where(armVec==app)[0]
        index_list.append(idx)
        index_list_len.append(len(idx))

    n_item_samples = np.zeros(n_items, dtype='int')
    n_item_rewards = np.zeros(n_items)
    regret = 0
    total_reward = 0
    n_rounds = 1

    # First trial for all arms in turn
    for item_idx in range(n_items):
        sel_idx = index_list[item_idx][0]
        reward = rewardVec[sel_idx] 
        n_item_samples[item_idx] += 1
        n_item_rewards[item_idx] += reward
        regret += max(actual_ctr) - actual_ctr[item_idx]
        
        total_reward += reward
        
        result = {}
        result['iteration'] = iteration
        result['round'] = n_rounds
        n_rounds += 1
        result['item_id'] = items[item_idx]
        result['reward'] = reward
        result['total_reward'] = total_reward
        result['avg_reward'] = total_reward * 1. / (n_rounds + 1)
        result['cumulative_regret'] = regret
        results.append(result)

    
    for n_rounds in range(n_items, rounds):
        exploration = 2 * np.log(n_rounds + 1) / n_item_samples
        exploration = np.power(exploration, 1/2)
        
        q = n_item_rewards + exploration
        
        item_idx = np.argmax(q)
        if (n_item_samples[item_idx] == index_list_len[item_idx]):
            break
        else:
            sel_idx = index_list[item_idx][n_item_samples[item_idx]] 
        
        reward = rewardVec[sel_idx] 
        n_item_samples[item_idx] += 1
        n_item_rewards[item_idx] += reward
        regret += max(actual_ctr) - actual_ctr[item_idx]
        total_reward += reward
        
        #alpha = 1./n_item_samples[item_idx]
        #n_item_rewards[item_idx] += alpha * (reward - n_item_rewards[item_idx])
        result = {}
        result['iteration'] = iteration
        result['round'] = n_rounds
        result['item_id'] = item_id
        result['reward'] = reward
        result['total_reward'] = total_reward
        result['avg_reward'] = total_reward * 1. / (n_rounds + 1)
        result['cumulative_regret'] = regret
        results.append(result)



#%%
ucb1_results = pd.DataFrame(ucb1_df(n_iterations=10,
                                    rounds=100000))

ucb1_results.to_csv('/Users/huiwon/Desktop/Recommendation/bandit/experiment/final_experiment/rounds/avazu_ucb1.csv')

#%%
