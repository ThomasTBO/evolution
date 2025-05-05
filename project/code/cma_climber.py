
from environnement import *
import cma
import json
from gif import create_gif
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
   
if __name__ == "__main__":
    
    climber0 = np.array([
    [3, 3, 3, 3, 3],
    [0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0],
    [0, 0, 4, 0, 0],
    [3, 3, 3, 3, 3]
    ])


    #PARAMETRES
    nb_sim = 7
    os.makedirs(f"project/solutions/ClimberEvol/Simu{nb_sim}", exist_ok=True)
    
    iterations_morpho = 2 # Number of iterations for the morpho evolution
    morpho_popsize = 10 # Population size for morpho evolution
    
    cma_gen_counter = 10 # Number of generations for CMA-ES
    cma_popsize = 10 # Population size for CMA-ES
    cma_max_steps = 100 # Number of steps for CMA-ES
    cma_sigma0 = 3 # Initial standard deviation for CMA-ES
    nb_elites = 3 # Proportion of elites to keep
    tournament_size = 3 # Size of the tournament for selection

    cma_gen_counter_final = 10 # Number of generations for final CMA-ES
    cma_popsize_final = 100 # Population size for final CMA-ES
    cma_max_steps_final = 200 # Number of steps for final CMA-ES 
    cma_sigma0_final = 10 # Initial standard deviation for final CMA-ES

    cma_gen_counter_final_2 = 100 # Number of generations for final CMA-ES
    cma_popsize_final_2 = 10 # Population size for final CMA-ES
    cma_max_steps_final_2 = 200 # Number of steps for final CMA-ES 
    cma_sigma0_final_2 = 1 # Initial standard deviation for final CMA-ES

    proba_mutate_elites = 8/25 # Probability of mutation for elites
    proba_mutate_tournament = 8/25 # Probability of mutation for tournament selection
    proba_mutate_inital = 15/25 # Probability of mutation for initial morpho

    
    previous_best = climber0
    previous_best_fitness = 0 
    new_gen = [morpho.mutate_climber(climber0, proba_mutate_inital) for _ in range(morpho_popsize-1)] + [climber0]
     # Initial population of morphologies
    robots_memory = {}
    for i in range(iterations_morpho): 
        pop = new_gen.copy()
        new_gen.clear() 
        new_gen = [previous_best]

        #Evaluation
        for robot in pop:
            robot_key = tuple(map(tuple, robot)) #Une morpho = une clé

            if robot_key not in robots_memory:
                genes, fitness, cfg = run_cma_par(robot, gen_counter=cma_gen_counter, max_steps=cma_max_steps, popsize=cma_popsize, sigma0=cma_sigma0)
                robots_memory[robot_key] = (genes, fitness, cfg) 

            # else : 
            #     genes, fitness, cfg = run_cma_par(robot, gen_counter=cma_gen_counter, max_steps=cma_max_steps, popsize=cma_popsize, sigma0=cma_sigma0)
            #     if fitness > robots_memory[robot_key][1]:
            #         robots_memory[robot_key] = (genes, fitness, cfg) 

        #Selection elite & mutations
        best_robot_key, (best_trained, best_fitness, best_cfg) = max(robots_memory.items(), key=lambda x: x[1][1])
        best_robot = np.array(best_robot_key)
        new_gen = [morpho.mutate_climber(best_robot, probability = proba_mutate_elites) for _ in range(nb_elites)] # Mutate the best robots to create new ones
        

        #Selection tournament & mutations
        while len(new_gen) < morpho_popsize:
            rng = np.random.default_rng()
            tournament = rng.choice(len(pop), size=tournament_size)
            robots = [pop[i] for i in tournament] 
            fitness = [robots_memory[tuple(map(tuple, robot))][1] for robot in robots]
            winner = robots[np.argmin(fitness)] # Select the best robot from the tournament
            new_gen.append(morpho.mutate_climber(winner, proba_mutate_tournament)) # Mutate the winner to create a new robot
        
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
    
        

    # Final training

    best_robot_key, (best_trained, best_fitness, best_cfg) = max(robots_memory.items(), key=lambda x: x[1][1])
    best_robot = np.array(best_robot_key)  # Convertir la clé (tuple) en matrice NumPy

    genes, fitness, cfg, max_fitness = run_cma_par(best_robot, gen_counter=cma_gen_counter_final, max_steps=cma_max_steps_final, popsize=cma_popsize_final, sigma0=cma_sigma0_final, genes=robots_memory[best_robot_key][0], show_fitness=True)
    robots_memory[best_robot_key] = (genes, fitness, cfg) 

    genes, fitness, cfg, max_fitness = run_cma_par(best_robot, gen_counter=cma_gen_counter_final, max_steps=cma_max_steps_final, popsize=cma_popsize_final, sigma0=cma_sigma0_final_2, genes=robots_memory[best_robot_key][0], show_fitness=True)
    robots_memory[best_robot_key] = (genes, fitness, cfg) 


    plt.plot(range(1, cma_gen_counter_final+1), max_fitness)
    print(f"Final training:")
    print(f"  Best fitness: {fitness}")
    print(f"  Best morphology:\n{best_robot}")
    plt.plot(range(1, cma_gen_counter_final+1), max_fitness)
    plt.savefig(f"project/solutions/ClimberEvol/Simu{nb_sim}/ClimberEvolution.png")
    
    
    name = f"ClimberEvol/Simu{nb_sim}/ClimberFinal"
    save_solution_cma(genes, fitness, cfg, name="project/solutions/" + name + ".json")
    create_gif_cma(name= name, max_steps=cma_max_steps_final)
    plt.show()
        
        
        # best_robot_key, (best_trained, fitness, best_cfg) = sorted_robots[0]
        # best = np.array(best_robot_key)  # Convertir la clé (tuple) en matrice NumPy

        # print(f"Iteration {i + 1}:")
        # print(f"  Best fitness: {fitness}")
        # print(f"  Best solution:\n{best}")

        # # Sauvegarder la meilleure solution
        # name = "WalkerEvo1_" + str(i + 1)
        # save_solution_cma(best_trained, fitness, best_cfg, name="project/solutions/MorphoEvol/" + name + ".json")
        # create_gif_cma(name="MorphoEvol/"+ name)
        

#MEMOIRE SIMULATION

# # #PARAMETRES
#     nb_sim = 6
#     os.makedirs(f"project/solutions/ClimberEvol/Simu{nb_sim}", exist_ok=True)
    
#     iterations_morpho = 5 # Number of iterations for the morpho evolution
#     morpho_popsize = 5 # Population size for morpho evolution
    
#     cma_gen_counter = 10 # Number of generations for CMA-ES
#     cma_popsize = 10 # Population size for CMA-ES
#     cma_max_steps = 100 # Number of steps for CMA-ES
#     cma_sigma0 = 2 # Initial standard deviation for CMA-ES

#     nb_elites = 3 # Proportion of elites to keep
#     tournament_size = 2 # Size of the tournament for selection

#     cma_gen_counter_final = 100 # Number of generations for final CMA-ES
#     cma_popsize_final = 20 # Population size for final CMA-ES
#     cma_max_steps_final = 500 # Number of steps for final CMA-ES 
#     cma_sigma0_final = 1 # Initial standard deviation for final CMA-ES

#     proba_mutate_elites = 5/25 # Probability of mutation for elites
#     proba_mutate_tournament = 5/25 # Probability of mutation for tournament selection
#     proba_mutate_inital = 15/25 # Probability of mutation for initial morpho
    