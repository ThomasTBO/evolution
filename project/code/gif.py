import json
import time
import imageio

from evolution import *


def create_gif(path):
    
    def load_solution(name=path):
        with open(name, "r") as f:
            cfg = json.load(f)
        cfg["robot"] = np.array(cfg["robot"])
        cfg["genes"] = np.array(cfg["genes"])
        a = Agent(Network, cfg, genes=cfg["genes"])
        a.fitness = cfg["fitness"]
        return a

    a = load_solution(name=path)
    cfg = a.config
    env = make_env(cfg["env_name"], robot=cfg["robot"], render_mode="rgb_array")
    env.metadata.update({'render_modes': ["rgb_array"]})

    a.fitness, imgs, steps = evaluate(a, env, render=True)
    env.close()
    print(a.fitness)
    
    # Save the images as a gif
    imageio.mimsave(f'project/solutions/solution.gif', imgs, duration=(1/50.0))

#create_gif(path="project/solutions/Walker0.json")