
from environnement import *
import cma
import json
from gif import create_gif
from imports import *
from torch import sigmoid
from gif_cma import create_gif_cma

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
def evaluate_env(env, agent, horizon = 500):
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
    return - value
    
    
    
if __name__ == "__main__":
   
    walker3 = np.array([
    [3, 3, 3, 3, 3],
    [3, 4, 0, 4, 3],
    [3, 0, 0, 4, 3],
    [3, 3, 0, 4, 3],
    [3, 3, 0, 4, 3]
    ])
    
    walker4 = np.array([
    [0, 4, 4, 4, 0],
    [4, 0, 3, 0, 4],
    [4, 3, 1, 3, 4],
    [4, 0, 3, 0, 4],
    [0, 4, 4, 4, 0]
    ])

    walker5 = np.array([
    [3, 3, 3, 3, 3],
    [3, 0, 0, 0, 3],
    [3, 0, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3]
    ])

    walker6 = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3]
    ])

    walker7 = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 0, 0, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3]
    ])
    

    config = {
        "env_name": "Walker-v0",
        "robot": walker5,
        "generations": 50, # To change: increase!
        "lambda": 10,
        "max_steps": 500, # to change to 500
        }

    cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs




    env = EvoGymEnv(cfg["env_name"], robot=cfg["robot"])

    ex_agent = Agent(Network, cfg)


    es = cma.CMAEvolutionStrategy(
        x0=ex_agent.genes,  # Initial mean (e.g., 2D search space)
        sigma0 = 1,  # Initial standard deviation
        inopts={'popsize': 20, 'verb_disp': 1}  # Options (e.g., population size, verbosity)
    )


    gen_counter = 0
    max_fitnesses = []

    for generation in range(config["generations"]):
        solutions = es.ask()
        
        # agent = Agent(Network, cfg)
        # nb_evals = 10
        # tasks = [evaluate_env.remote(env, agent,  horizon=1000) for _ in range(nb_evals)]
        # results = ray.get(tasks)    

        tasks = [evaluate_env.remote(env, Agent(Network, cfg, genes=genes), horizon=cfg["max_steps"]) for genes in solutions]
        fitnesses = ray.get(tasks)

        es.tell(solutions, fitnesses)
        if es.stop() : break

        gen_counter +=1
        max_fitnesses.append(-es.result.fbest)

    print("Best fitness:", -es.result.fbest)
    plt.plot(range(1,gen_counter+1),max_fitnesses)
    plt.show()

    def save_solution_cma(genes, fitness, cfg, name="project/solutions/solution.json"):
        save_cfg = {}
        for i in ["env_name", "robot", "n_in", "h_size", "n_out"]:
            assert i in cfg, f"{i} not in config"
            save_cfg[i] = cfg[i]
        save_cfg["robot"] = cfg["robot"].tolist()
        save_cfg["genes"] = genes.tolist()
        save_cfg["fitness"] = float(fitness)
        # save
        with open(name, "w") as f:
            json.dump(save_cfg, f)
        return save_cfg


    name = "WalkerCMA7"
    save_solution_cma(es.result.xbest, -es.result.fbest, cfg, name="project/solutions/" + name + ".json")
    create_gif_cma(name = name)



