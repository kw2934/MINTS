import numpy as np
import cvxpy as cp

### MINTS for multi-armed bandits (including Lipschitz bandits)
class MINTS_MAB:
    def __init__(self, K, prior = None, reward_model = 'Gaussian', reward_std = 0.5, distance_matrix = None, Lipschitz_const = None, epsilon = 1e-5, seed = None):
        # K: number of arms
        # prior: prior distribution of the optimum, a numpy array of K nonnegative entries, with at least one positive entry
        # reward_model: 'Gaussian' or 'Bernoulli'
        # reward_std: standard deviation proxy of the sub-Gaussian reward
        # distance_matrix: (for Lipschitz bandits) pairwise distances between arms, a numpy array of size (K, K) with nonnegative entries
        # Lipschitz_const: (for Lipschitz bandits) a bound on the Lipschitz constant, a nonnegative number        
        # epsilon: (for Bernoulli rewards) to avoid numerical issues, we assume the expected rewards belong to [epsilon, 1 - epsilon]
        # seed: random seed

        self.K = K
        self.reward_model = reward_model
        self.sigma = reward_std
        self.decisions = [] # record the indices of chosen arms
        self.feedbacks = [] # record the observed rewards

        # set the random seed
        self.rng = np.random.default_rng(seed = seed)
        
        ## initialization
        # prior and posterior distributions of the optimal decision
        if prior is None:
            self.prior = np.ones(self.K) / self.K
        else:
            self.prior = prior / np.sum(prior)
        self.posterior = self.prior.copy()

        self.num_of_rounds = 0 # record the number of rounds
        self.counts = np.zeros(self.K) # how many times each arm is pulled
        self.rewards_sum = np.zeros(self.K) # cumulative reward corresponding to each arm

        ## set up the convex program for computing the profile likelihood
        # unknown parameters: expected rewards
        self.theta = cp.Variable(self.K)

        # optimality constraints
        # placeholder: if the j-th arm is assumed to be optimal, we set self.idx_opt_placeholder = e_j
        self.idx_opt_placeholder = cp.Parameter(self.K)
        self.constraints = [ self.idx_opt_placeholder @ self.theta >= self.theta ]

        # smoothness constraints (for Lipschitz bandits)
        if distance_matrix is not None:
            self.constraints += [cp.abs(self.theta[i] - self.theta[j]) <= Lipschitz_const * distance_matrix[i, j] for i in range(self.K) for j in range(self.K)]        

        # model-specific objective function and constraints
        if self.reward_model == 'Bernoulli':
            # placeholder: cumulative reward for each arm, divided by the number of rounds
            self.normalized_rewards_placeholder = cp.Parameter(self.K, nonneg = True)
            self.normalized_zero_rewards_placeholder = cp.Parameter(self.K, nonneg = True)
            
            # objective function: log-likelihood divided by the number of rounds
            self.normalized_log_likelihood = self.normalized_rewards_placeholder @ cp.log(self.theta) + self.normalized_zero_rewards_placeholder @ cp.log(1 - self.theta)
            
            # constraint: expected rewards belong to [0, 1]
            self.constraints += [self.theta >= 0 + epsilon, self.theta <= 1 - epsilon]

        elif self.reward_model == 'Gaussian':   
            # placeholder: cumulative reward for each arm, divided by the number of rounds
            self.normalized_rewards_placeholder = cp.Parameter(self.K)            

            # placeholder: number of times each price is chosen, divided by the number of rounds
            self.weights_placeholder = cp.Parameter(self.K, nonneg = True)
            
            # objective function: log-likelihood (up to an additive constant) divided by the number of rounds
            self.normalized_log_likelihood = - ( self.weights_placeholder @ cp.square(self.theta) - 2 * self.normalized_rewards_placeholder @ self.theta ) / (2 * (self.sigma ** 2))

        # problem setup
        self.objective = cp.Maximize(self.normalized_log_likelihood)
        self.problem = cp.Problem(self.objective, self.constraints)


    ### sample a price
    def sample_decision(self):
        self.new_idx = self.rng.choice(self.K, p = self.posterior, size = 1)[0]
        return self.new_idx


    ### update belief after observing the demand given the chosen price
    def update_belief(self, feedback):
        # record data
        self.decisions.append(int(self.new_idx))
        self.feedbacks.append(feedback)

        # update sufficient statistics
        self.counts[self.new_idx] += 1
        self.rewards_sum[self.new_idx] += feedback
        self.num_of_rounds += 1

        ## compute profile likelihood
        # assign values to parameters and solve the program
        self.normalized_rewards_placeholder.value = self.rewards_sum / self.num_of_rounds
        
        if self.reward_model == 'Bernoulli':        
            self.normalized_zero_rewards_placeholder.value = self.counts / self.num_of_rounds - self.normalized_rewards_placeholder.value
        elif self.reward_model == 'Gaussian':
            self.weights_placeholder.value = self.counts / self.num_of_rounds

        log_profile_likelihood = np.empty(self.K)
        for j in range(self.K):
            tmp = np.zeros(self.K)
            tmp[j] = 1
            self.idx_opt_placeholder.value = tmp
            self.problem.solve()
            log_profile_likelihood[j] = self.problem.value * self.num_of_rounds

        log_profile_likelihood -= np.max(log_profile_likelihood)
        log_profile_likelihood = np.clip(log_profile_likelihood, a_min = -15, a_max = float('inf')) # clipping
        tmp = np.exp(log_profile_likelihood) * self.prior
        self.posterior = tmp / np.sum(tmp)



### MINTS for dynamic pricing
class MINTS_dynamic_pricing:
    def __init__(self, prices, prior = None, demand_model = 'Bernoulli', demand_std = 1, Lipschitz_const = 1, epsilon = 1e-5, seed = None):
        # prices: a numpy array of K candidate prices, sorted in ascending order
        # prior: prior distribution of the optimum, a numpy array of K nonnegative entries, with at least one positive entry
        # demand_model: conditional distribution of the demand given a price, 'Bernoulli' or 'Gaussian'
        # demand_std: standard deviation of the Gaussian demand distribution (ignored if model == 'Bernoulli')
        # Lipschitz_const: Lipschitz constant of the CDF of latent valuation
        # epsilon: (for Bernoulli demands) to avoid numerical issues, we assume the expected demands belong to [epsilon, 1 - epsilon]
        # seed: random seed 

        self.prices = prices # candidate prices
        self.K = len(prices) # number of candidate prices
        self.demand_model = demand_model
        self.sigma = demand_std
        self.decisions = [] # record chosen price indices
        self.feedbacks = [] # record observed demands

        # set the random seed
        self.rng = np.random.default_rng(seed = seed)
        
        ## initialization
        # prior and posterior distributions of the optimal decision
        if prior is None:
            self.prior = np.ones(self.K) / self.K
        else:
            self.prior = prior / np.sum(prior)
        self.posterior = self.prior.copy()
        
        self.num_of_rounds = 0 # record the number of rounds
        self.counts = np.zeros(self.K) # how many times each price is used
        self.demands_sum = np.zeros(self.K) # cumulative demand corresponding to each price

        ## set up the convex program for computing the profile likelihood
        # unknown parameter: expected demand given each price
        self.theta = cp.Variable(self.K)
        
        # monotonicity constraints
        self.constraints = [self.theta[1:] <= self.theta[:-1]]

        # smoothness constraints
        self.constraints += [self.theta[:-1] - self.theta[1:] <= Lipschitz_const * (self.prices[1:] - self.prices[:-1])]

        # optimality constraints
        # placeholder: if the j-th price is assumed to be optimal, we set self.price_opt_placeholder = price[j] * e_j
        self.price_opt_placeholder = cp.Parameter(self.K)
        self.constraints += [ self.price_opt_placeholder @ self.theta >= cp.multiply(self.prices, self.theta) ]

        # placeholder: cumulative demand for each price, divided by the number of rounds
        self.normalized_demands_placeholder = cp.Parameter(self.K, nonneg = True)

        # model-specific objective function and constraints
        if self.demand_model == 'Bernoulli':
            # placeholder
            self.normalized_missed_demands_placeholder = cp.Parameter(self.K, nonneg = True)
            
            # objective function: log-likelihood divided by the number of rounds
            self.normalized_log_likelihood = self.normalized_demands_placeholder @ cp.log(self.theta) + self.normalized_missed_demands_placeholder @ cp.log(1 - self.theta)
            
            # constraint: expected demand belongs to [0, 1]
            self.constraints += [self.theta[-1] >= 0 + epsilon, self.theta[0] <= 1 - epsilon]

        elif self.demand_model == 'Gaussian':            
            # placeholder: number of times each price is chosen, divided by the number of rounds
            self.weights_placeholder = cp.Parameter(self.K, nonneg = True)
            
            # objective function: log-likelihood (up to an additive constant) divided by the number of rounds
            self.normalized_log_likelihood = - ( self.weights_placeholder @ cp.square(self.theta) - 2 * self.normalized_demands_placeholder @ self.theta ) / (2 * (self.sigma ** 2))

        # problem setup
        self.objective = cp.Maximize(self.normalized_log_likelihood)
        self.problem = cp.Problem(self.objective, self.constraints)

            
    ### sample a price (return the index)
    def sample_decision(self):
        self.price_new_idx = self.rng.choice(self.K, p = self.posterior, size = 1)[0]
        return self.price_new_idx
    

    ### update belief after observing the demand given the chosen price
    def update_belief(self, feedback):
        # record data
        self.decisions.append(int(self.price_new_idx))
        self.feedbacks.append(feedback)

        # update sufficient statistics
        self.counts[self.price_new_idx] += 1
        self.demands_sum[self.price_new_idx] += feedback
        self.num_of_rounds += 1

        ## compute profile likelihood
        # assign values to parameters and solve the program
        self.normalized_demands_placeholder.value = self.demands_sum / self.num_of_rounds
        
        if self.demand_model == 'Bernoulli':        
            self.normalized_missed_demands_placeholder.value = self.counts / self.num_of_rounds - self.normalized_demands_placeholder.value
        elif self.demand_model == 'Gaussian':
            self.weights_placeholder.value = self.counts / self.num_of_rounds

        log_profile_likelihood = np.empty(self.K)
        for j in range(self.K):
            tmp = np.zeros(self.K)
            tmp[j] = self.prices[j]
            self.price_opt_placeholder.value = tmp
            self.problem.solve()
            log_profile_likelihood[j] = self.problem.value * self.num_of_rounds

        log_profile_likelihood -= np.max(log_profile_likelihood)
        log_profile_likelihood = np.clip(log_profile_likelihood, a_min = -15, a_max = float('inf')) # clipping
        tmp = np.exp(log_profile_likelihood) * self.prior
        self.posterior = tmp / np.sum(tmp)

   
############################
#### Baseline algorithms: Thompson sampling and UCB1

### Thompson sampling for multi-armed bandits
class TS:
    def __init__(self, K, reward_model = 'Gaussian', prior_means = None, prior_vars = None, reward_std = 1, prior_alphas = None, prior_betas = None, seed = None):
        # K: number of arms
        # seed: random seed
        # reward_model: 'Gaussian' or 'Bernoulli'

        # For Gaussian reward model: one Gaussian prior for each mean reward
        # prior_means: the mean parameters in the Gaussian priors, a numpy array of K entries
        # prior_vars: the variance parameters in the Gaussian priors, a numpy array of K positive entries
        # reward_std: standard deviation of the reward, a positive number

        # For Bernoulli reward model: one Beta prior for each mean reward
        # prior_alphas: the alpha parameters in the Beta priors, a numpy array of K positive entries
        # prior_betas: the beta parameters in the Beta priors, a numpy array of K positive entries

        self.K = K
        self.reward_model = reward_model
        self.decisions = [] # record the indices of chosen arms
        self.feedbacks = [] # record the observed rewards
        
        ## prior distributions
        if reward_model == 'Gaussian':
            self.inverse_variance = 1 / reward_std ** 2
            if prior_means is None:
                self.prior_means = np.zeros(self.K)
            else:
                self.prior_means = prior_means
            if prior_vars is None:
                self.prior_vars = np.ones(self.K)
            else:
                self.prior_vars = prior_vars    
            self.posterior_means = self.prior_means.copy()  
            self.posterior_vars = self.prior_vars.copy()  

            self.posterior_inverse_vars = 1 / self.prior_vars
            self.posterior_ratio_parameters = self.prior_means / self.prior_vars  

        if reward_model == 'Bernoulli':
            if prior_alphas is None:
                self.prior_alphas = np.ones(self.K)
            else:
                self.prior_alphas = prior_alphas
            if prior_betas is None:
                self.prior_betas = np.ones(self.K)
            else:
                self.prior_betas = prior_betas  
                
            self.posterior_alphas = self.prior_alphas.copy()  
            self.posterior_betas = self.prior_betas.copy()  

        # set the random seed
        self.rng = np.random.default_rng(seed = seed)


    ### sample a decision
    def sample_decision(self):
        if self.reward_model == 'Gaussian':
            theta = self.rng.normal(loc = self.posterior_means, scale = np.sqrt(self.posterior_vars))
        if self.reward_model == 'Bernoulli':
            theta = self.rng.beta(self.posterior_alphas, self.posterior_betas)
        self.new_idx = np.argmax(theta)
        return self.new_idx
    

    ### update belief after observing the reward given the decision
    def update_belief(self, feedback):
        # record data
        x = int(self.new_idx)
        self.decisions.append(x)
        self.feedbacks.append(feedback)

        # update sufficient statistics        
        if self.reward_model == 'Gaussian':
            self.posterior_inverse_vars[x] += self.inverse_variance
            self.posterior_vars[x] = 1 / self.posterior_inverse_vars[x]
            
            self.posterior_ratio_parameters[x] += feedback * self.inverse_variance            
            self.posterior_means[x] = self.posterior_ratio_parameters[x] * self.posterior_vars[x]
            
        if self.reward_model == 'Bernoulli':
            self.posterior_alphas[x] += feedback
            self.posterior_betas[x] += 1 - feedback


### UCB algorithms for multi-armed bandits
class UCB:
    def __init__(self, K, T, method = 'UCB1', reward_std = 0.5):
        # K: number of arms
        # T: total number of rounds
        # method: 'UCB1' or 'UCB_optimal'
        # UCB1: Auer, Cesa-Bianchi and Fischer (2002)
        # UCB-optimal: Chapter 8.1 Algorithm 6 in Lattimore and Szepesvári (2020), an asymptotically optimal UCB algorithm

        # reward_std: standard deviation proxy of the sub-Gaussian reward, a positive number
        # if each arm's reward is supported on an interval [a, b], set reward_std = (b - a) / 2
        # delta: exceptional probability parameter

        self.K = K
        self.T = T
        self.method = method
        self.sigma = reward_std
        self.num_of_rounds = 0 # record the number of rounds
        self.counts = np.zeros(self.K) # how many times each arm is pulled
        self.rewards_sum = np.zeros(self.K) # cumulative reward corresponding to each arm
        self.decisions = [] # record the indices of chosen arms
        self.feedbacks = [] # record the observed rewards


    ### sample a decision
    def sample_decision(self):
        if self.num_of_rounds < self.K:
            self.new_idx = self.num_of_rounds        
        else:
            means = self.rewards_sum / self.counts
            if self.method == 'UCB1':
                bonuses = 2 * self.sigma * np.sqrt( 2 * np.log(self.T) / self.counts )
            elif self.method == 'UCB_optimal':
                f = 1 + (self.num_of_rounds + 1) * ( np.log(self.num_of_rounds + 1) ** 2 )
                bonuses = 2 * self.sigma * np.sqrt( 2 * np.log(f) / self.counts )
            self.new_idx = np.argmax(means + bonuses)
        return self.new_idx
    

    ### update belief after observing the reward given the decision
    def update_belief(self, feedback):
        # record data
        x = int(self.new_idx)
        self.decisions.append(x)
        self.feedbacks.append(feedback)

        # update sufficient statistics        
        self.counts[x] += 1
        self.rewards_sum[x] += feedback
        self.num_of_rounds += 1


