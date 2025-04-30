import json
import time
import imageio

from evolution import *

def create_gif_cma(cfg, name):
    path = "project/solutions/" + name + ".json"
    
    with open(path, "r") as f:
        solution = json.load(f)
    solution["robot"] = np.array(solution["robot"])
    solution["genes"] = np.array(solution["genes"])
    a = Agent(Network, cfg, genes=solution["genes"])
    a.fitness = solution["fitness"]
    print(a.fitness)
    
    env = make_env(cfg["env_name"], robot=cfg["robot"], render_mode="rgb_array")
    env.metadata.update({'render_modes': ["rgb_array"]})
    
    a.fitness, imgs = evaluate(a, env, render=True)
    env.close()
    print(a.fitness)
    
    # Save the images as a gif
    imageio.mimsave(f'project/solutions/' + name + '.gif', imgs, duration=(1/50.0))


config = {
    "env_name": "Walker-v0",
    "robot": walker,
    "generations": 50, # To change: increase!
    "lambda": 10,
    "max_steps": 100, # to change to 500
    }

if __name__ == "__main__":
    cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
    cfg = {**config, **cfg} # Merge configs
    create_gif_cma(cfg, name="WalkerCMA")
    