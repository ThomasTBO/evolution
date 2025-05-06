
from environnement import *
import cma
import json
from project.code.codes_inutiles.gif import create_gif
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
    
    
    
# if __name__ == "__main__":
   
#     walker3 = np.array([
#     [3, 3, 3, 3, 3],
#     [3, 4, 0, 4, 3],
#     [3, 0, 0, 4, 3],
#     [3, 3, 0, 4, 3],
#     [3, 3, 0, 4, 3]
#     ])
    
#     walker4 = np.array([
#     [0, 4, 4, 4, 0],
#     [4, 0, 3, 0, 4],
#     [4, 3, 1, 3, 4],
#     [4, 0, 3, 0, 4],
#     [0, 4, 4, 4, 0]
#     ])

#     walker5 = np.array([
#     [3, 3, 3, 3, 3],
#     [3, 0, 0, 0, 3],
#     [3, 0, 0, 3, 3],
#     [3, 3, 0, 3, 3],
#     [3, 3, 0, 3, 3]
#     ])

#     walker6 = np.array([
#     [3, 3, 3, 3, 3],
#     [3, 3, 0, 3, 3],
#     [3, 3, 0, 3, 3],
#     [3, 3, 0, 3, 3],
#     [3, 3, 0, 3, 3]
#     ])

#     walker7 = np.array([
#     [3, 3, 3, 3, 3],
#     [3, 3, 0, 0, 3],
#     [3, 3, 0, 3, 3],
#     [3, 3, 0, 3, 3],
#     [3, 3, 0, 3, 3]
#     ])
    
#     walker8 = np.array([
#     [3, 3, 3, 3, 3],
#     [3, 3, 3, 3, 3],
#     [3, 3, 0, 3, 3],
#     [3, 3, 0, 3, 3],
#     [3, 0, 0, 0, 3]
#     ])
    
#     thrower1= np.array([
#     [3, 0, 0, 0, 0],
#     [3, 3, 3, 3, 3],
#     [0, 0, 0, 0, 3],
#     [0, 0, 0, 0, 3],
#     [3, 3, 3, 3, 3]
#     ])

#     config = {
#         "env_name": "Thrower-v0",
#         "robot": thrower1,
#         "generations": 80, # To change: increase!
#         "lambda": 10,
#         "max_steps": 500, # to change to 500
#         }

#     cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
#     cfg = {**config, **cfg} # Merge configs


#     env = EvoGymEnv(cfg["env_name"], robot=cfg["robot"])

#     ex_agent = Agent(Network, cfg)


#     es = cma.CMAEvolutionStrategy(
#         x0=ex_agent.genes,  # Initial mean (e.g., 2D search space)
#         sigma0 = 1,  # Initial standard deviation
#         inopts={'popsize': 20, 'verb_disp': 1}  # Options (e.g., population size, verbosity)
#     )


#     gen_counter = 0
#     max_fitnesses = []

#     for generation in range(config["generations"]):
#         solutions = es.ask()
        
#         # agent = Agent(Network, cfg)
#         # nb_evals = 10
#         # tasks = [evaluate_env.remote(env, agent,  horizon=1000) for _ in range(nb_evals)]
#         # results = ray.get(tasks)    

#         tasks = [evaluate_env.remote(env, Agent(Network, cfg, genes=genes), horizon=cfg["max_steps"]) for genes in solutions]
#         fitnesses = ray.get(tasks)

#         es.tell(solutions, fitnesses)
#         if es.stop() : break

#         gen_counter +=1
#         max_fitnesses.append(-es.result.fbest)

#     print("Best fitness:", -es.result.fbest)
#     plt.plot(range(1,gen_counter+1),max_fitnesses)
#     plt.show()

#     

#     name = "ThrowerCMA2"
#     save_solution_cma(es.result.xbest, -es.result.fbest, cfg, name="project/solutions/" + name + ".json")
#     create_gif_cma(name = name)

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
    global seed
    config = {
        "env_name": "Thrower-v0",
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
        inopts={'popsize': popsize, 'verb_disp': 1, 'seed' : seed}, # Options (e.g., population size, verbosity)
       
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
    
    thrower0 = np.array([
    [4, 0, 0, 0, 0],
    [4, 0, 0, 0, 4],
    [4, 4, 4, 4, 4],
    [0, 0, 0, 4, 4],
    [4, 4, 4, 4, 4]
    ])

    thrower1 = np.array([
    [4, 4, 0, 0, 0],
    [4, 4, 0, 0, 4],
    [4, 4, 4, 4, 4],
    [0, 4, 4, 4, 4],
    [4, 4, 4, 4, 4]
    ])


   

    #PARAMETRES
    nb_sim = 73
    os.makedirs(f"project/solutions/ThrowerEvol/Simu{nb_sim}", exist_ok=True)

    seed = 1

    iterations_morpho = 1 # Number of iterations for the morpho evolution
    morpho_popsize = 106 # Population size for morpho evolution
    
    cma_gen_counter = 15 # Number of generations for CMA-ES
    cma_popsize = 5 # Population size for CMA-ES
    cma_max_steps = 200 # Number of steps for CMA-ES
    cma_sigma0 = 1 # Initial standard deviation for CMA-ES

    nb_elites = 5 # Proportion of elites to keep
    tournament_size = 6 # Size of the tournament for selection

    cma_gen_counter_final = 100 # Number of generations for final CMA-ES
    cma_popsize_final = 20 # Population size for final CMA-ES
    cma_max_steps_final = 200 # Number of steps for final CMA-ES 
    cma_sigma0_final = 0.5 # Initial standard deviation for final CMA-ES

    proba_mutate_elites = 10/25 # Probability of mutation for elites
    proba_mutate_tournament = 9/25 # Probability of mutation for tournament selection
    proba_mutate_inital = proba_mutate_elites
    
    previous_best = thrower0
    previous_best_fitness = 0 
    new_gen = [morpho.mutate(thrower0, proba_mutate_inital) for _ in range(morpho_popsize-1)] + [thrower0] 
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
        new_gen = [morpho.mutate(best_robot, probability = proba_mutate_elites) for _ in range(nb_elites)] # Mutate the best robots to create new ones
        

        #Selection tournament & mutations
        while len(new_gen) < morpho_popsize:
            rng = np.random.default_rng()
            tournament = rng.choice(len(pop), size=tournament_size)
            robots = [pop[i] for i in tournament] 
            fitness = [robots_memory[tuple(map(tuple, robot))][1] for robot in robots]
            winner = robots[np.argmin(fitness)] # Select the best robot from the tournament
            new_gen.append(morpho.mutate(winner, proba_mutate_tournament)) # Mutate the winner to create a new robot
        
        print(f"Iteration {i + 1}:")
        print(f"  Best fitness: {best_fitness}")
        print(f"  Best morphology:\n{best_robot}")
        if np.array_equal(previous_best, best_robot) and best_fitness == previous_best_fitness:
            print("Same morphology as previous best.")
        else:
            name = f"ThrowerEvol/Simu{nb_sim}/ThrowerIter{i+1}"
            save_solution_cma(best_trained, best_fitness, best_cfg, name="project/solutions/" + name + ".json")
            create_gif_cma(name= name)
            previous_best = best_robot
            previous_best_fitness = best_fitness
    
        

    #Entrainement CMA-ES sur la meilleure morpho
    best_robot_key, (best_trained, best_fitness, best_cfg) = max(robots_memory.items(), key=lambda x: x[1][1])
    best_robot = np.array(best_robot_key)  # Convertir la clé (tuple) en matrice NumPy
    genes, fitness, cfg, list_fitness = run_cma_par(best_robot, gen_counter=cma_gen_counter_final, max_steps=cma_max_steps_final, popsize=cma_popsize_final, sigma0=cma_sigma0_final, genes=robots_memory[best_robot_key][0], show_fitness=True)
    print(f"Final training:")
    print(f"  Best fitness: {fitness}")
    print(f"  Best morphology:\n{best_robot}")

    
    
    name = f"ThrowerEvol/Simu{nb_sim}/ThrowerFinal"
    save_solution_cma(genes, fitness, cfg, name="project/solutions/" + name + ".json")
    create_gif_cma(name= name, max_steps=cma_max_steps_final)
    plt.plot(range(1, cma_gen_counter_final+1), list_fitness)
    plt.savefig(f"project/solutions/ThrowerEvol/Simu{nb_sim}/ThrowerFinal.png")
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
#SIM 3
#   #PARAMETRES
#     nb_sim = 3
    
#     iterations_morpho = 3 # Number of iterations for the morpho evolution
#     morpho_popsize = 30 # Population size for morpho evolution
    
#     cma_gen_counter = 20 # Number of generations for CMA-ES
#     cma_popsize = 5 # Population size for CMA-ES
#     cma_max_steps = 200 # Number of steps for CMA-ES
#     cma_sigma0 = 1 # Initial standard deviation for CMA-ES

#     nb_elites = 5 # Proportion of elites to keep
#     tournament_size = 7 # Size of the tournament for selection

#     cma_gen_counter_final = 30 # Number of generations for final CMA-ES
#     cma_popsize_final = 30 # Population size for final CMA-ES
#     cma_max_steps_final = 200 # Number of steps for final CMA-ES 
#     cma_sigma0_final = 0.5 # Initial standard deviation for final CMA-ES

#     proba_mutate_elites = 7/25 # Probability of mutation for elites
#     proba_mutate_tournament = 7/25 # Probability of mutation for tournament selection

  #PARAMETRES
    # nb_sim = 4
    
    # iterations_morpho = 6 # Number of iterations for the morpho evolution
    # morpho_popsize = 30 # Population size for morpho evolution
    
    # cma_gen_counter = 10 # Number of generations for CMA-ES
    # cma_popsize = 5 # Population size for CMA-ES
    # cma_max_steps = 200 # Number of steps for CMA-ES
    # cma_sigma0 = 1 # Initial standard deviation for CMA-ES

    # nb_elites = 5 # Proportion of elites to keep
    # tournament_size = 7 # Size of the tournament for selection

    # cma_gen_counter_final = 67 # Number of generations for final CMA-ES
    # cma_popsize_final = 15 # Population size for final CMA-ES
    # cma_max_steps_final = 200 # Number of steps for final CMA-ES 
    # cma_sigma0_final = 0.5 # Initial standard deviation for final CMA-ES

    # proba_mutate_elites = 7/25 # Probability of mutation for elites
    # proba_mutate_tournament = 7/25 # Probability of mutation for tournament selection

    #  nb_sim = 5
    
    # iterations_morpho = 6 # Number of iterations for the morpho evolution
    # morpho_popsize = 30 # Population size for morpho evolution
    
    # cma_gen_counter = 10 # Number of generations for CMA-ES
    # cma_popsize = 5 # Population size for CMA-ES
    # cma_max_steps = 200 # Number of steps for CMA-ES
    # cma_sigma0 = 1 # Initial standard deviation for CMA-ES

    # nb_elites = 15 # Proportion of elites to keep
    # tournament_size = 4 # Size of the tournament for selection

    # cma_gen_counter_final = 67 # Number of generations for final CMA-ES
    # cma_popsize_final = 15 # Population size for final CMA-ES
    # cma_max_steps_final = 200 # Number of steps for final CMA-ES 
    # cma_sigma0_final = 0.5 # Initial standard deviation for final CMA-ES

    # proba_mutate_elites = 5/25 # Probability of mutation for elites
    # proba_mutate_tournament = 8/25 # Probability of mutation for tournament selection

# nb_sim = 6
#     os.makedirs(f"project/solutions/ThrowerEvol/Simu{nb_sim}", exist_ok=True)
    
#     iterations_morpho = 6 # Number of iterations for the morpho evolution
#     morpho_popsize = 30 # Population size for morpho evolution
    
#     cma_gen_counter = 10 # Number of generations for CMA-ES
#     cma_popsize = 5 # Population size for CMA-ES
#     cma_max_steps = 200 # Number of steps for CMA-ES
#     cma_sigma0 = 1 # Initial standard deviation for CMA-ES

#     nb_elites = 5 # Proportion of elites to keep
#     tournament_size = 6 # Size of the tournament for selection

#     cma_gen_counter_final = 67 # Number of generations for final CMA-ES
#     cma_popsize_final = 15 # Population size for final CMA-ES
#     cma_max_steps_final = 200 # Number of steps for final CMA-ES 
#     cma_sigma0_final = 0.5 # Initial standard deviation for final CMA-ES

#     proba_mutate_elites = 7/25 # Probability of mutation for elites
#     proba_mutate_tournament = 9/25 # Probability of mutation for tournament selection
    

    #    nb_sim = 7
    # os.makedirs(f"project/solutions/ThrowerEvol/Simu{nb_sim}", exist_ok=True)
    
    # iterations_morpho = 1 # Number of iterations for the morpho evolution
    # morpho_popsize = 150 # Population size for morpho evolution
    
    # cma_gen_counter = 15 # Number of generations for CMA-ES
    # cma_popsize = 5 # Population size for CMA-ES
    # cma_max_steps = 200 # Number of steps for CMA-ES
    # cma_sigma0 = 1 # Initial standard deviation for CMA-ES

    # nb_elites = 5 # Proportion of elites to keep
    # tournament_size = 6 # Size of the tournament for selection

    # cma_gen_counter_final = 67  # Number of generations for final CMA-ES
    # cma_popsize_final = 15 # Population size for final CMA-ES
    # cma_max_steps_final = 200 # Number of steps for final CMA-ES 
    # cma_sigma0_final = 0.5 # Initial standard deviation for final CMA-ES

    # proba_mutate_elites = 10/25 # Probability of mutation for elites
    # proba_mutate_tournament = 9/25 # Probability of mutation for tournament selection
    
    # #PARAMETRES
    # nb_sim = 7.1
    # os.makedirs(f"project/solutions/ThrowerEvol/Simu{nb_sim}", exist_ok=True)
    
    # iterations_morpho = 1 # Number of iterations for the morpho evolution
    # morpho_popsize = 106 # Population size for morpho evolution
    
    # cma_gen_counter = 15 # Number of generations for CMA-ES
    # cma_popsize = 5 # Population size for CMA-ES
    # cma_max_steps = 200 # Number of steps for CMA-ES
    # cma_sigma0 = 1 # Initial standard deviation for CMA-ES

    # nb_elites = 5 # Proportion of elites to keep
    # tournament_size = 6 # Size of the tournament for selection

    # cma_gen_counter_final = 100 # Number of generations for final CMA-ES
    # cma_popsize_final = 20 # Population size for final CMA-ES
    # cma_max_steps_final = 200 # Number of steps for final CMA-ES 
    # cma_sigma0_final = 0.5 # Initial standard deviation for final CMA-ES

    # proba_mutate_elites = 10/25 # Probability of mutation for elites
    # proba_mutate_tournament = 9/25 # Probability of mutation for tournament selection
    
    #  #PARAMETRES
    # nb_sim = 7
    # os.makedirs(f"project/solutions/ThrowerEvol/Simu{nb_sim}", exist_ok=True)
    
    # iterations_morpho = 3 # Number of iterations for the morpho evolution
    # morpho_popsize = 40 # Population size for morpho evolution
    
    # cma_gen_counter = 10 # Number of generations for CMA-ES
    # cma_popsize = 5 # Population size for CMA-ES
    # cma_max_steps = 250 # Number of steps for CMA-ES
    # cma_sigma0 = 1 # Initial standard deviation for CMA-ES

    # nb_elites = 14 # Proportion of elites to keep
    # tournament_size = 15 # Size of the tournament for selection

    # cma_gen_counter_final = 200 # Number of generations for final CMA-ES
    # cma_popsize_final = 20 # Population size for final CMA-ES
    # cma_max_steps_final = 500 # Number of steps for final CMA-ES 
    # cma_sigma0_final = 0.3 # Initial standard deviation for final CMA-ES

    # proba_mutate_elites = 10/25 # Probability of mutation for elites
    # proba_mutate_tournament = 10/25 # Probability of mutation for tournament selection
    
    #  #PARAMETRES
    # nb_sim = 8
    # os.makedirs(f"project/solutions/ThrowerEvol/Simu{nb_sim}", exist_ok=True)
    
    # iterations_morpho = 4 # Number of iterations for the morpho evolution
    # morpho_popsize = 15 # Population size for morpho evolution
    
    # cma_gen_counter = 33 # Number of generations for CMA-ES
    # cma_popsize = 10 # Population size for CMA-ES
    # cma_max_steps = 250 # Number of steps for CMA-ES
    # cma_sigma0 = 2 # Initial standard deviation for CMA-ES

    # nb_elites = 5 # Proportion of elites to keep
    # tournament_size = 5 # Size of the tournament for selection

    # cma_gen_counter_final = 200 # Number of generations for final CMA-ES
    # cma_popsize_final = 20 # Population size for final CMA-ES
    # cma_max_steps_final = 500 # Number of steps for final CMA-ES 
    # cma_sigma0_final = 1 # Initial standard deviation for final CMA-ES

    # proba_mutate_elites = 7/25 # Probability of mutation for elites
    # proba_mutate_tournament = 9/25 # Probability of mutation for tournament selection
    # proba_mutate_inital = 10/25 # Probability of mutation for initial morpho
    
    