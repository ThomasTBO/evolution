import ray 
import gymnasium as gym
import evogym.envs
import numpy as np
import time
from agents import Agent, Network
from environnement import get_cfg, get_full_connectivity
from imports import *
from torch import sigmoid

class Network(nn.Module):
    def __init__(self, n_in, h_size, n_out):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, n_out)
 
        self.n_out = n_out

    def reset(self):
        pass
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = sigmoid(x)+0.6
        return x
    
class Agent:
    def __init__(self, Net, config, genes = None):
        self.config = config
        self.Net = Net
        self.model = None
        self.fitness = None

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.make_network()
        if genes is not None:
            self.genes = genes

    def __repr__(self):  # pragma: no cover
        return f"Agent {self.model} > fitness={self.fitness}"

    def __str__(self):  # pragma: no cover
        return self.__repr__()

    def make_network(self):
        n_in = self.config["n_in"]
        h_size = self.config["h_size"]
        n_out = self.config["n_out"]
        self.model = self.Net(n_in, h_size, n_out).to(self.device).double()
        return self

    @property
    def genes(self):
        if self.model is None:
            return None
        with torch.no_grad():
            params = self.model.parameters()
            vec = torch.nn.utils.parameters_to_vector(params)
        return vec.cpu().double().numpy()

    @genes.setter
    def genes(self, params):
        if self.model is None:
            self.make_network()
        assert len(params) == len(
            self.genes), "Genome size does not fit the network size"
        if np.isnan(params).any():
            raise
        a = torch.tensor(params, device=self.device)
        torch.nn.utils.vector_to_parameters(a, self.model.parameters())
        self.model = self.model.to(self.device).double()
        self.fitness = None
        return self

    def mutate_ga(self):
        genes = self.genes
        n = len(genes)
        f = np.random.choice([False, True], size=n, p=[1/n, 1-1/n])
        
        new_genes = np.empty(n)
        new_genes[f] = genes[f]
        noise = np.random.randn(n-sum(f))
        new_genes[~f] = noise
        return new_genes

    def act(self, obs):
        # continuous actions
        with torch.no_grad():
            x = torch.tensor(obs).double().unsqueeze(0).to(self.device)
            actions = self.model(x).cpu().detach().numpy()
        return actions

class EvoGymEnv:
    def __init__(self, env_name, robot):
        import gymnasium as gym
        import evogym.envs
        self.env = gym.make(env_name,body=robot)
        self.env_name = env_name
        self.robot = robot
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
   

    def __reduce__(self):
        deserializer = self.__class__
        serialized_data = (self.env_name, self.robot)
        return deserializer, serialized_data
    
    def reset(self):
        """
        Reset the environment and return the initial observation.
        """
        return self.env.reset()
    
    def step(self, action):
        """
        Take a step in the environment with the given action.
        """
        obs, reward, done, trunc,  info = self.env.step(action)
        return obs, reward, done, trunc, info
    

@ray.remote
def evaluate_env(env, agent, horizon = 1000):
    """
    Evaluate the environment for a given number of steps.
    """
    obs,i = env.reset()
    done = False
    value = 0
    for _ in range(horizon):
        # action = env.action_space.sample()
        action = agent.act(obs)  # Use the agent to get the action
        obs, reward, done, trunc,  info = env.step(action)
        value += reward
    return value
    
    
    
if __name__ == "__main__":
   
    walker = np.array([
        [3, 3, 3, 3, 3],
        [3, 3, 3, 0, 3],
        [3, 3, 0, 3, 3],
        [3, 3, 0, 3, 3],
        [3, 3, 0, 3, 3]
        ])
    env = EvoGymEnv("Walker-v0", walker)
    config = {
    "env_name": "Walker-v0",
    "robot": walker,
    "generations": 20, # To change: increase!
    "lambda": 10,
    "max_steps": 500, # to change to 500
    }

    cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs

    #sans ray : 
    #env = make_env(cfg["env_name"], robot=cfg["robot"])


    #Avec ray :
    env = EvoGymEnv(cfg["env_name"], robot=cfg["robot"])

    agent = Agent(Network, cfg)

    nb_evals = 10
    #  Check time series
    t_0 = time.time()
    for _ in range(nb_evals):
        task = evaluate_env.remote(env, agent, horizon=1000)
        result = ray.get(task)
    t_1 = time.time()
    print(f"Time taken for {nb_evals} evaluations in series: {t_1 - t_0:.2f} seconds")

    # Check time parallel
    t_0 = time.time()
    tasks = [evaluate_env.remote(env, agent,  horizon=1000) for _ in range(nb_evals)]
    results = ray.get(tasks)
    t_1 = time.time()
    print(f"Time taken for {nb_evals} evaluations in parallel: {t_1 - t_0:.2f} seconds")