
from evolution import *
import cma
import json
from gif import create_gif

config = {
    "env_name": "Walker-v0",
    "robot": walker,
    "generations": 20, # To change: increase!
    "lambda": 10,
    "max_steps": 500, # to change to 500
    }

cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
cfg = {**config, **cfg} # Merge configs
    
env = make_env(cfg["env_name"], robot=cfg["robot"])

ex_agent = Agent(Network, cfg)


es = cma.CMAEvolutionStrategy(
    x0=ex_agent.genes,  # Initial mean (e.g., 2D search space)
    sigma0 = 0.1,  # Initial standard deviation
    inopts={'popsize': 20, 'verb_disp': 1}  # Options (e.g., population size, verbosity)
)


gen_counter = 0
max_fitnesses = []

for generation in range(config["generations"]):
    solutions = es.ask()
    fitnesses = [- evaluate(Agent(Network, cfg, genes=genes), env, max_steps=cfg["max_steps"]) for genes in solutions]
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


name = "WalkerCMA"
save_solution_cma(es.result.xbest, -es.result.fbest, cfg, name="project/solutions/" + name + ".json")
#create_gif_cma(cfg, name = name)





# a = one_plus_lambda(config)

# solutions = es.ask()
# fitnesses = [objective_function(solution) for solution in solutions]

# def objective(a, cfg):
#     env = make_env(cfg["env_name"], robot=cfg["robot"])
#     fit = evaluate(a, env, max_steps=cfg["max_steps"])
#     env.close()
#     return -fit

# if __name__ == "__main__":
#     config = {
#     "env_name": "Walker-v0",
#     "robot": walker,
#     "generations": 10, # To change: increase!
#     "lambda": 10,
#     "max_steps": 100, # to change to 500
#     }

#     optimizer = CMA(mean=np.zeros(1), sigma=1)

#     for generation in range(50):
#         solutions = []
#         for _ in range(optimizer.population_size):
#             x = optimizer.ask()
#             value = objective(x[0], x[1])
#             solutions.append((x, value))
#             print(f"#{generation} {value} (x1={x[0]}, x2 = {x[1]})")
#         optimizer.tell(solutions)


# def run_cmaes(config):
#     # Initial guess and standard deviation
#     initial_guess = [0] * config["n_genes"]
#     sigma = 0.5  # Standard deviation for mutation

#     # CMA-ES optimization
#     result = cma.fmin(
#         lambda genes: objective_function(genes, config["env_name"], config["robot"], config["max_steps"]),
#         initial_guess,
#         sigma,
#         options={
#             "popsize": config["lambda"],  # Population size
#             "maxiter": config["generations"],  # Number of generations
#         },
#     )

#     # Best solution
#     best_genes = result[0]
#     best_fitness = -result[1]  # Negate to get the actual fitness
#     print(f"Best fitness: {best_fitness}")
#     return best_genes


# if __name__ == "__main__":
#     config = {
#         "env_name": "Walker-v0",
#         "robot": walker,
#         "generations": 50,
#         "lambda": 10,
#         "max_steps": 500,
#         "n_genes": 100,  # Adjust based on your agent's genome size
#     }

#     best_genes = run_cmaes(config)
#     print("Best genes:", best_genes)

#     env = make_env(config["env_name"], robot=config["robot"])
#     agent = Agent(Network, {"n_in": env.observation_space.shape[0], "h_size": 32, "n_out": env.action_space.shape[0]}, genes=best_genes)
#     reward = evaluate(agent, env, render=True)
#     print(f"Final reward: {reward}")
#     env.close()