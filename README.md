# A Minimalist Bayesian Framework for Stochastic Optimization

Paper: (https://arxiv.org/abs/2509.07030).

## Demonstration

See `demo.ipynb` for a demonstration of the MINimalist Thompson Sampling (MINTS) algorithm. The experiment uses dynamic pricing as a test bed.

### Problem setup

Consider selling a product to a sequence of potential buyers. Let $D_0 = \varnothing$ be an empty dataset. At time $t \geq 1$,
- we choose a price $x_t$ from a feasible set, based on historical data $D_{t-1}$;
- a potential buyer with valuation $v_t$ sees the price and makes a purchase if $v_t \geq x_t$;
- we observe a binary demand $\phi_t = \mathbf{1} ( v_t \geq x_t )$ and update our dataset to $D_t = D_{t-1} \cup \lbrace (x_t, \phi_t)  \rbrace$.

Assume that
- the feasible prices are $p_1 < \cdots < p_K$;
- $\lbrace v_t \rbrace_{t=1}^{\infty}$ are independent samples from an unknown distribution $\rho$, whose density is bounded by a known constant $L$.

Each price $p$ is associated with an expected revenue $R(p) = p \cdot \mathbb{P}_{v \sim \rho} ( v \geq p )$. The goal is to maximize this objective:

$$ \max_{j \in [K] } R(p_j). $$

The above model has $K$ unknown parameters: $\theta_j = \mathbb{P}_{v \sim \rho} ( v \geq p_j )$ for $j \in [K]$. They satisfy the following constraints:
- Monotonicity: $1 \geq \theta_1 \geq \cdots \geq \theta_K \geq 0$;
- Lipschitz continuity: $\theta_j - \theta_{j+1} \leq L ( p_{j+1} - p_{j} )$ for $j \in [K - 1]$.

The MINTS algorithm maintains a distributional estimate of the optimal price to capture the uncertainties. Starting from a prior distribution, such as the uniform distribution over $\lbrace p_j \rbrace_{j=1}^K$, MINTS conducts Bayesian-type belief updates as data comes in and nicely handles the constraints on model parameters. At each time $t$, the new price $x_t$ is randomly sampled from a generalized posterior distribution of the optimal price given data $D_{t-1}$. 

In contrast, fully Bayesian approaches (e.g., Thompson sampling) require a prior for all the unknown parameters $\lbrace \theta_j \rbrace_{j=1}^K$, which is difficult to design given the constraints. Computing and sampling from the posterior distribution are also hard.


### Experiment

Our demonstration uses $K = 5$, $p_j = (2 + j) / 10$, $L = 1$ and $\rho = \mathcal{U} [0, 1]$. The MINTS algorithm has access to $\lbrace p_j \rbrace_{j=1}^K$ and $L$ but not $\rho$. Below we plot how the generalized posterior distribution of the optimal price evolves over time, starting from the uniform prior.

<p align="center">
    <img src="posterior_snapshots.png" alt="Demonstration" width="1000" height="800" />
</p>

To evaluate the performance of MINTS, we compute its cumulative regret $\sum_{i=1}^t [ R^* - R(x_t) ]$ over time. We also compare it with Thompson sampling for unstructured multi-armed bandits (using Gaussian likelihood and prior), which targets mean reward parameters $\lbrace R(p_j) \rbrace_{j=1}^K$ and ignores the constraints on $\lbrace \theta_j \rbrace_{j=1}^K$.

<p align="center">
    <img src="regrets.png" alt="regrets" width="500" height="400" />
</p>

The curves show that MINTS significantly benefits from the incorporation of structural constraints.


## Experiments in the paper

To reproduce the numerical results in Section 5, please refer to the folder `Experiments`. 
- Use `pricing_experiments.ipynb` and `MAB_experiments.ipynb` to run the experiments. The main functions are written in `experiments.py`.
- The outcomes are stored in the compressed folder named `Data.zip`. For statistical analysis and visualization, unzip the data folder and use `pricing_plots.ipynb` and `MAB_plots.ipynb`.


## Citation
```
@article{Wang2025,
  title={A Minimalist Bayesian Framework for Stochastic Optimization},
  author={Wang, Kaizheng},
  journal={arXiv preprint arXiv:2509.07030},
  year={2025}
}
```

