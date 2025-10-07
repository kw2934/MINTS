import numpy as np
import sys
sys.path.append('/content/drive/MyDrive/Colab Notebooks/MINTS')
from MINTS import *

### Multi-armed bandits with Bernoulli and Gaussian rewards
def simulation_MAB(T, list_seeds, list_methods, file_path, theta, model = 'Bernoulli', reward_std = 0.5):
    # T: total number of rounds
    # list_seeds: a list of random seeds
    # list_methods: a list of methods. candidates: 'MINTS_Bernoulli', 'MINTS_Gaussian', 'TS_Bernoulli', 'TS_Gaussian', 'UCB1', 'UCB_optimal'
    # file_path: path to save the results
    # theta: mean rewards of K arms
    # model: distribution family of the reward, 'Bernoulli' or 'Gaussian'
    # reward_std: (for Gaussian model) standard deviation of the rewards
    
    import pickle, os

    # record the results
    regrets = dict() # regrets[seed][method] gives the regret array of a specific method under a specific random seed
    decisions = dict() # decisions[seed][method] gives the chosen arm indices array of a specific method under a specific random seed  
    rewards = dict() # rewards[seed][method] gives the feedbacks array of a specific method under a specific random seed
    rewards_raw = dict() # rewards_raw[seed] gives the raw rewards matrix (T * K) under a specific random seed

    # check if partial results exist
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            tmp = pickle.load(f)
        regrets = tmp['regrets']
        decisions = tmp['decisions']
        rewards = tmp['rewards']
        rewards_raw = tmp['rewards_raw']


    # run the experiments
    print(f'Start experiments for multi-armed bandits. {len(list_seeds)} random seeds in total.')

    for (idx, seed) in enumerate(list_seeds):
        if seed in regrets.keys():
            print(f'Task {idx} (random seed = {seed}) already finished before.')
            continue

        print(f'Start: task index = {idx}, random seed = {seed}.')

        tmp = dict()
        test = simulation_MAB_single(T = T, theta = theta, model = model, reward_std = reward_std, seed = seed)
        test.run(list_methods = list_methods)

        regrets[seed] = test.regrets
        decisions[seed] = test.decisions
        rewards[seed] = test.rewards
        rewards_raw[seed] = test.rewards_raw

        results = dict()
        results['regrets'] = regrets
        results['decisions'] = decisions
        results['rewards'] = rewards
        results['rewards_raw'] = rewards_raw       
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f'Task {idx} finished.')

    print('All tasks finished.')


# single run of multi-armed bandits
class simulation_MAB_single:
    def __init__(self, T, theta, model = 'Bernoulli', reward_std = 0.5, seed = 1000):
        self.T = T
        self.theta = theta
        self.reward_std = reward_std
        self.seed = seed
        self.rng = np.random.default_rng(seed) # set random seed

        # generate potential rewards for all arms
        self.K = len(theta)
        self.model = model
        if model == 'Bernoulli':
            self.rewards_raw = self.rng.binomial(n = 1, p = theta, size = (self.T, self.K))
        elif model == 'Gaussian':
            self.rewards_raw = self.rng.normal(loc = theta, scale = reward_std, size = (self.T, self.K))


    def run(self, list_methods):
        self.regrets = dict()
        self.decisions = dict()
        self.rewards = dict()

        for method in list_methods:
            self.regrets[method] = np.zeros(self.T)

        seed_new = 2 * self.seed
        self.alg = dict()
        if 'MINTS_Bernoulli' in self.regrets.keys():
            self.alg['MINTS_Bernoulli'] = MINTS_MAB(K = self.K, prior = None, reward_model = 'Bernoulli', seed = seed_new)
        if 'MINTS_Gaussian' in self.regrets.keys():
            self.alg['MINTS_Gaussian'] = MINTS_MAB(K = self.K, prior = None, reward_model = 'Gaussian', reward_std = self.reward_std, seed = seed_new)
        if 'TS_Bernoulli' in self.regrets.keys():
            self.alg['TS_Bernoulli'] = TS(K = self.K, reward_model = 'Bernoulli', seed = seed_new)
        if 'TS_Gaussian' in self.regrets.keys():
            self.alg['TS_Gaussian'] = TS(K = self.K, reward_model = 'Gaussian', reward_std = self.reward_std, seed = seed_new)
        if 'UCB1' in self.regrets.keys():
            self.alg['UCB1'] = UCB(K = self.K, T = self.T,  method = 'UCB1', reward_std = self.reward_std)
        if 'UCB_optimal' in self.regrets.keys():
            self.alg['UCB_optimal'] = UCB(K = self.K, T = self.T,  method = 'UCB_optimal', reward_std = self.reward_std)

        
        # online experiments
        theta_max = np.max(self.theta)
        for t in range(self.T):
            for method in list_methods:
                idx = self.alg[method].sample_decision() # sample an arm
                self.alg[method].update_belief(feedback = self.rewards_raw[t, idx])
                self.regrets[method][t] = theta_max - self.theta[idx]


        # compute regret, record decisions and rewards        
        for method in list_methods:
            self.regrets[method] = np.cumsum(self.regrets[method])
            self.decisions[method] = self.alg[method].decisions
            self.rewards[method] = self.alg[method].feedbacks



                    
### Dynamic pricing with Bernoulli demand and Uniform[0, 1] latent valuation

def simulation_dynamic_pricing(T, prices, list_seeds, list_methods, file_path, demand_std = 0.5, Lipschitz_const = 1):
    # T: total number of rounds
    # prices: a list of candidate prices, sorted in ascending order
    # list_seeds: a list of random seeds
    # list_methods: a list of methods. Candidates: 'MINTS_pricing_Bernoulli', 'MINTS_pricing_Gaussian', 'TS', 'UCB1', 'UCB_optimal'
    # file_path: path to save the results

    import pickle, os

    # record the results
    regrets = dict() # regrets[seed][method] gives the regret array of a specific method under a specific random seed
    price_indices = dict() # price_indices[seed][method] gives the chosen price indices array of a specific method under a specific random seed  
    demands = dict() # demands[seed][method] gives the feedbacks array of a specific method under a specific random seed
    valuations = dict() # valuations[seed] gives the latent valuations array (T * 1) under a specific random seed

    # check if partial results exist
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            tmp = pickle.load(f)
        regrets = tmp['regrets']
        price_indices = tmp['price indices']
        demands = tmp['demands']
        valuations = tmp['valuations']


    # run the experiments
    print(f'Start experiments for dynamic pricing. {len(list_seeds)} random seeds in total.')

    for (idx, seed) in enumerate(list_seeds):
        if seed in regrets.keys():
            print(f'Task {idx} (random seed = {seed}) already finished before.')
            continue

        print(f'Start: task index = {idx}, random seed = {seed}.')

        tmp = dict()
        test = simulation_dynamic_pricing_single(T = T, prices = prices, seed = seed)
        test.run(list_methods = list_methods, demand_std = demand_std, Lipschitz_const = Lipschitz_const)

        regrets[seed] = test.regrets
        price_indices[seed] = test.price_indices
        demands[seed] = test.demands
        valuations[seed] = test.V

        results = dict()
        results['regrets'] = regrets
        results['price indices'] = price_indices
        results['demands'] = demands        
        results['valuations'] = valuations
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f'Task {idx} finished.')

    print('All tasks finished.')


# single run of dynamic pricing
class simulation_dynamic_pricing_single:
    def __init__(self, T, prices, seed = 1000):
        self.T = T
        self.prices = prices
        self.K = len(prices)
        self.seed = seed
        self.rng = np.random.default_rng(seed) # set random seed
        self.V = self.rng.random(T) # latent valuations

    def run(self, list_methods, demand_std = 0.5, Lipschitz_const = 1):
        self.price_indices = dict() # record chosen price indices
        self.demands = dict() # record demands
        self.regrets = dict() # record regrets

        for method in list_methods:
            self.regrets[method] = np.zeros(self.T)

        seed_new = 2 * self.seed
        self.alg = dict()
        if 'MINTS_pricing_Bernoulli' in self.regrets.keys():
            self.alg['MINTS_pricing_Bernoulli'] = MINTS_dynamic_pricing(self.prices, prior = None, demand_model = 'Bernoulli', Lipschitz_const = Lipschitz_const, seed = seed_new)
        if 'MINTS_pricing_Gaussian' in self.regrets.keys():
            self.alg['MINTS_pricing_Gaussian'] = MINTS_dynamic_pricing(self.prices, prior = None, demand_model = 'Gaussian', demand_std = demand_std, Lipschitz_const = Lipschitz_const, seed = seed_new)
        if 'TS' in self.regrets.keys():
            self.alg['TS'] = TS(K = self.K, reward_model = 'Gaussian', reward_std = demand_std, seed = seed_new)
            self.demands['TS'] = []
        if 'UCB1' in self.regrets.keys():
            self.alg['UCB1'] = UCB(K = self.K, T = self.T,  method = 'UCB1', reward_std = demand_std)
            self.demands['UCB1'] = []
        
        # online experiments
        R = self.prices * (1 - self.prices) # expected revenues
        R_max = np.max(R) # maximum expected revenue
        for t in range(self.T):
            for method in list_methods:
                idx = self.alg[method].sample_decision() # sample a price index
                demand = int(self.V[t] >= self.prices[idx]) # observe the demand
                self.regrets[method][t] = R_max - R[idx]

                # update the belief
                if method in {'MINTS_pricing_Bernoulli', 'MINTS_pricing_Gaussian'}:
                    self.alg[method].update_belief(feedback = demand)
                else:
                    self.alg[method].update_belief(feedback = self.prices[idx] * demand)
                    self.demands[method].append(demand) # record demands for TS and UCB1

        # compute regret, record decisions and feedbacks   
        for method in list_methods:
            self.regrets[method] = np.cumsum(self.regrets[method])
            self.price_indices[method] = self.alg[method].decisions
            if method in {'MINTS_pricing_Bernoulli', 'MINTS_pricing_Gaussian'}:
                self.demands[method] = self.alg[method].feedbacks



### Merge results from different task groups

def merge_results(folder_path, experiment_type, num_groups):
    # experiment_type: 'pricing' or 'MAB'
    import pickle
    results = dict()
    for id_group in range(1, num_groups + 1):
        file_path_i = folder_path + 'results_{}_{}_{}.pkl'.format(experiment_type, num_groups, id_group)
        with open(file_path_i, 'rb') as f:
            results_i = pickle.load(f)
        results.update(results_i)

    file_path = folder_path + 'results_{}_all_seeds.pkl'.format(experiment_type)
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)


### summary statistics

def summarize_results(file_path, benchmark = None):
    # file_path: path to the saved results
    # benchmark: name of a benchmark method (for computing regret ratios) or None
    
    import pickle
    with open(file_path, 'rb') as f:
        tmp = pickle.load(f)
    results_raw = tmp['regrets'] # results_raw[seed][method] gives the regret array of a specific method under a specific random seed

    list_seeds = list( results_raw.keys() ) # a list of random seeds
    list_methods = list( results_raw[list_seeds[0]].keys() ) #`a list of methods`
    T = len(results_raw[list_seeds[0]][list_methods[0]]) # total number of rounds
    
    regret = dict()
    for method in list_methods:
        regret[method] = np.zeros((len(list_seeds), T + 1))    
        for (i, seed) in enumerate(list_seeds):
            regret[method][i, 1:] = results_raw[seed][method]
    
    # compute the ratio to the benchmark method
    if benchmark is not None:
        ratio = dict()
        for method in list_methods:
            ratio[method] = regret[method] / np.clip(regret[benchmark], a_min = 1e-5, a_max = None)                                

    summary_stats = dict()
    for method in list_methods:
        summary_stats[method] = dict()
        summary_stats[method]['mean'] = np.mean(regret[method], axis = 0)
        summary_stats[method]['se'] = np.std(regret[method], axis = 0) / np.sqrt(len(list_seeds))
        if benchmark is not None:
            summary_stats[method]['mean_ratio'] = np.mean(ratio[method], axis = 0)
            summary_stats[method]['se_ratio'] = np.std(ratio[method], axis = 0) / np.sqrt(len(list_seeds))
    
    return summary_stats
