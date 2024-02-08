#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings(action='ignore')
from numba import jit

#%%
@jit
def ucb_tuned_df(n_iterations, rounds):
    main_df = np.genfromtxt('/Users/huiwon/Desktop/Recommendation/bandit/experiment/data/avazu/data/top_app_category.csv', delimiter=',', dtype=str, encoding='UTF-8')
    main_df = np.delete(main_df, 0, axis=0)
    main_df = np.delete(main_df, [0,4], axis=1)
    main_df[:,0] = main_df[:,0].astype(int)

    item_col_name = 2

    items = np.unique(main_df[:,item_col_name])
    n_items = len(items)

    actual_ctr = []
    for app in items:
        actual_ctr.append(np.mean(main_df[np.where(main_df[:,2]==app)][:,0].astype(int)))


    results = []
    for iteration in tqdm(range(n_iterations)):
        
        np.random.seed(iteration)
        sample_df = np.random.permutation(main_df)
        
        n_item_samples = np.zeros(n_items)
        n_item_rewards = np.zeros(n_items)
        
        regret = 0
        
        total_reward = 0
        
        for n_rounds in range(n_items):
            
            item_idx = n_rounds
            item_id = items[item_idx]
            
            check_df = sample_df[np.where(sample_df[:,2] == item_id)][0]
            
            reward = int(check_df[0])
            
            delete_row = np.min(np.where(sample_df[:,2] == item_id))
            
            sample_df = np.delete(sample_df, delete_row, axis=0)
            
            n_item_samples[item_idx] += 1
            n_item_rewards[item_idx] += reward
            
            regret += max(actual_ctr) - actual_ctr[item_idx]
            
            total_reward += reward
            
            result = {}
            result['iteration'] = iteration
            result['round'] = n_rounds
            result['item_id'] = item_id
            result['reward'] = reward
            result['total_reward'] = total_reward
            result['avg_reward'] = total_reward * 1. / (n_rounds + 1)
            result['cumulative_regret'] = regret
            results.append(result)
        
        for n_rounds in range(n_items, rounds):
            
            var_value = (n_item_rewards) - (n_item_rewards**2)
            var_value += np.power(2 * np.log(n_rounds + 1) / n_item_samples, 1/2)
            var_value = np.where(var_value > 0.25, 0.25, var_value)
            exploration = (np.log(n_rounds + 1) / n_item_samples) * var_value
            exploration = np.power(exploration, 1/2)
            
            q = n_item_rewards + exploration
            
            item_idx = np.argmax(q)
            item_id = items[item_idx]
            
            if sample_df[np.where(sample_df[:,2] == item_id)].shape[0] > 0:
                check_df = sample_df[np.where(sample_df[:,2] == item_id)][0]
            else:
                break
            
            reward = int(check_df[0])
            
            delete_row = np.min(np.where(sample_df[:,2] == item_id))
            
            sample_df = np.delete(sample_df, delete_row, axis=0)
            n_item_samples[item_idx] += 1
            
            alpha = 1./n_item_samples[item_idx]
            n_item_rewards[item_idx] += alpha * (reward - n_item_rewards[item_idx])
            
            regret += max(actual_ctr) - actual_ctr[item_idx]
            
            total_reward += reward
            
            result = {}
            result['iteration'] = iteration
            result['round'] = n_rounds
            result['item_id'] = item_id
            result['reward'] = reward
            result['total_reward'] = total_reward
            result['avg_reward'] = total_reward * 1. / (n_rounds + 1)
            result['cumulative_regret'] = regret

            results.append(result)
    
    return results

#%%
ucb_tuned_results = pd.DataFrame(ucb_tuned_df(n_iterations=10,
                                                rounds=100000))

ucb_tuned_results.to_csv('/Users/huiwon/Desktop/Recommendation/bandit/experiment/final_experiment/rounds/avazu_ucb_tuned.csv')

#%%
