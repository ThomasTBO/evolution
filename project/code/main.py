from evolution import *
import json
from gif import create_gif

def save_solution(a, cfg, name="project/solutions/solution.json"):
    save_cfg = {}
    for i in ["env_name", "robot", "n_in", "h_size", "n_out"]:
        assert i in cfg, f"{i} not in config"
        save_cfg[i] = cfg[i]
    save_cfg["robot"] = cfg["robot"].tolist()
    save_cfg["genes"] = a.genes.tolist()
    save_cfg["fitness"] = float(a.fitness)
    # save
    with open(name, "w") as f:
        json.dump(save_cfg, f)
    return save_cfg


if __name__ == "__main__":

    config = {
    "env_name": "Walker-v0",
    "robot": walker,
    "generations": 10, # To change: increase!
    "lambda": 10,
    "max_steps": 100, # to change to 500
    }

    ray.init()

    a = one_plus_lambda(config)
    # a.fitness
    env = make_env(config["env_name"], robot=config["robot"])
    reward, steps = evaluate(a, env, render=False)
    print(f"Reward: {reward}")
    env.close()
   
    #np.save("project/solutions/Walker0.npy", a.genes)
    # cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
    # cfg = {**config, **cfg} # Merge configs
    # save_solution(a, cfg, name="project/solutions/Walker0.json")
    #create_gif(path="project/solutions/solution.json")
