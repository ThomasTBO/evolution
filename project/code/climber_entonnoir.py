from environnement import *
import cma
import json
from imports import *
from torch import sigmoid
from gif_cma import create_gif_cma
import os

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


import morpho

def run_cma_par(robot, gen_counter=40, max_steps=500, popsize=20, sigma0=1,genes=None, show_fitness=False):

    config = {
        "env_name": "Climber-v2",
        "robot": robot,
        "generations": gen_counter, # To change: increase!
        "lambda": 10,
        "max_steps": max_steps, # to change to 500
        }

    cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs
    env = EvoGymEnv(cfg["env_name"], robot=cfg["robot"])
    ex_agent = Agent(Network, cfg)
    if genes is None:
        genes = ex_agent.genes
    es = cma.CMAEvolutionStrategy(
        x0=genes,  # Initial mean (e.g., 2D search space)
        sigma0 = sigma0,  # Initial standard deviation
        inopts={'popsize': popsize, 'verb_disp': 1}  # Options (e.g., population size, verbosity)
    )
    gen_counter = 0
    max_fitnesses = []
    for generation in range(config["generations"]):
        solutions = es.ask()
        tasks = [evaluate_env.remote(env, Agent(Network, cfg, genes=genes), horizon=cfg["max_steps"]) for genes in solutions]
        fitnesses = ray.get(tasks)
        es.tell(solutions, fitnesses)
        if es.stop() : break
        gen_counter +=1
        max_fitnesses.append(-es.result.fbest)
    if show_fitness:
        return es.result.xbest, -es.result.fbest, cfg, max_fitnesses
    return es.result.xbest, -es.result.fbest, cfg

def evaluation(pop, cma_gen_counter, cma_max_steps, cma_popsize, cma_sigma0):
    global robots_memory
    for robot in pop:
        robot_key = tuple(map(tuple, robot)) #Une morpho = une clé

        if robot_key not in robots_memory:
            genes, fitness, cfg = run_cma_par(robot, gen_counter=cma_gen_counter, max_steps=cma_max_steps, popsize=cma_popsize, sigma0=cma_sigma0)
            robots_memory[robot_key] = (genes, fitness, cfg) 

        else : 
            genes, fitness, cfg = run_cma_par(robot, gen_counter=cma_gen_counter, max_steps=cma_max_steps, popsize=cma_popsize, sigma0=cma_sigma0, genes=robots_memory[robot_key][0])
            if fitness > robots_memory[robot_key][1]:
                robots_memory[robot_key] = (genes, fitness, cfg) 

if __name__ == "__main__":
    
    climber0 = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3],
    [3, 3, 3, 3, 3]
    ])


    #PARAMETRES
    nb_sim = 100
    os.makedirs(f"project/solutions/ClimberEvol/Simu{nb_sim}", exist_ok=True)
    
    # iterations_morpho = 3 # Number of iterations for the morpho evolution
    morpho_popsize= 4 # Population size for morpho evolution
    
    # cma_gen_counter = 10 # Number of generations for CMA-ES
    # cma_popsize = 5 # Population size for CMA-ES
    # cma_max_steps = 250 # Number of steps for CMA-ES
    # cma_sigma0 = 2 # Initial standard deviation for CMA-ES

    # nb_elites = 5 # Proportion of elites to keep
    # tournament_size = 5 # Size of the tournament for selection

    # cma_gen_counter_final = 100 # Number of generations for final CMA-ES
    # cma_popsize_final = 5 # Population size for final CMA-ES
    # cma_max_steps_final = 500 # Number of steps for final CMA-ES 
    # cma_sigma0_final = 1 # Initial standard deviation for final CMA-ES

    # proba_mutate_elites = 6/25 # Probability of mutation for elites
    # proba_mutate_tournament = 6/25 # Probability of mutation for tournament selection
    proba_mutate_inital = 15/25 # Probability of mutation for initial morpho
    
    
    previous_best = climber0
    previous_best_fitness = 0 
    new_gen = [morpho.mutate(climber0, proba_mutate_inital) for _ in range(morpho_popsize-1)] + [climber0] 
    robots_memory = {}

   
    pop = new_gen.copy()
    new_gen.clear() 
    new_gen = [previous_best]

    #Evaluation
    def cycle(nb_cma_gen, len_cma_pop, nb_gardes, cma_max_steps = 500, cma_sigma0 = 1, show_fitness = False):
        global pop
        global robots_memory
        evaluation(pop, nb_cma_gen, cma_max_steps, len_cma_pop, cma_sigma0)
        sorted_robots = sorted(robots_memory.items(), key=lambda x: x[1][1], reverse=True) # Sort the robots by fitness
        next_gen = sorted_robots[:nb_gardes]
        pop = [np.array(robot_key) for robot_key, _ in next_gen] # Keep the best robots for the next generation


    def printer(i):
        global robots_memory, nb_sim, previous_best, previous_best_fitness
        best_robot_key, (best_trained, best_fitness, best_cfg) = max(robots_memory.items(), key=lambda x: x[1][1])
        best_robot = np.array(best_robot_key)  # Convertir la clé (tuple) en matrice NumPy
        print(f"Iteration {i + 1}:")
        print(f"  Best fitness: {best_fitness}")
        print(f"  Best morphology:\n{best_robot}")
        if np.array_equal(previous_best, best_robot) and best_fitness == previous_best_fitness:
            print("Same morphology as previous best.")
        else:
            name = f"ClimberEvol/Simu{nb_sim}/Climber{i+1}"
            save_solution_cma(best_trained, best_fitness, best_cfg, name="project/solutions/" + name + ".json")
            create_gif_cma(name= name)
            previous_best = best_robot
            previous_best_fitness = best_fitness

    def saver():
        global nb_sim, robots_memory
        best_robot_key, (best_trained, best_fitness, best_cfg) = max(robots_memory.items(), key=lambda x: x[1][1])
        name = f"ClimberEvol/Simu{nb_sim}/ClimberFinal"
        save_solution_cma(best_trained, best_fitness, best_cfg, name="project/solutions/" + name + ".json")
        create_gif_cma(name= name, max_steps=500)

    #CYCLES ENTONNOIRS
    cycle(nb_gardes = 1, nb_cma_gen= 10,  len_cma_pop= 50, cma_max_steps = 100, cma_sigma0 = 5) 
    printer(0)
    # cycle(nb_gardes = 1, nb_cma_gen= 50, len_cma_pop = 15,cma_max_steps = 100, cma_sigma0 = 1.5) 
    # printer(1)
    
    gen_counter_final = 3
    popsize_final = 20
    gen_counter_final = 200
    max_steps_final = 300
    
    best_robot = pop[0]
    best_robot_key = tuple(map(tuple, best_robot)) #Une morpho = une clé
    genes, fitness, cfg , max_fitness = run_cma_par(best_robot, gen_counter=gen_counter_final, max_steps=max_steps_final, popsize=popsize_final, sigma0=1, genes=robots_memory[best_robot_key][0], show_fitness=True)
    robots_memory[best_robot_key] = (genes, fitness, cfg) 
    plt.plot(range(1, gen_counter_final+1), max_fitness)
    printer(1)
    saver()
    plt.savefig(f"project/solutions/ClimberEvol/Simu{nb_sim}/ClimberEvolution.png")
    plt.show()  

   


#MEMOIRE SIMULATION

#SIMU3
# cycle(nb_gardes = 5, nb_cma_gen= 10,  len_cma_pop= 5, cma_max_steps = 50, cma_sigma0 = 2) 
#     printer(0)
#     cycle(nb_gardes = 1, nb_cma_gen= 50, len_cma_pop = 15,cma_max_steps = 100, cma_sigma0 = 1.5) 
#     printer(1)
#     # cycle(nb_gardes = 1, nb_cma_gen= 40, len_cma_pop = 10,cma_max_steps = 300, cma_sigma0 = 1) 
#     #cycle(nb_gardes = 1, nb_cma_gen= 50, len_cma_pop = 10,cma_max_steps = 300, cma_sigma0 = 1) 

#     best_robot_key, (best_trained, best_fitness, best_cfg) = max(robots_memory.items(), key=lambda x: x[1][1])
#     best_robot = np.array(best_robot_key)  # Convertir la clé (tuple) en matrice NumPy
#     gen_counter_final = 3
#     popsize_final = 20
#     gen_counter_final = 200
#     max_steps_final = 300

#SIMU4
 #4 individus
    # cycle(nb_gardes = 1, nb_cma_gen= 10,  len_cma_pop= 50, cma_max_steps = 100, cma_sigma0 = 5) 
    # printer(0)
    # # cycle(nb_gardes = 1, nb_cma_gen= 50, len_cma_pop = 15,cma_max_steps = 100, cma_sigma0 = 1.5) 
    # # printer(1)
    
    # gen_counter_final = 3
    # popsize_final = 20
    # gen_counter_final = 200
    # max_steps_final = 300