import json
import imageio
from environnement import *
from agents import *
from imports import *


def create_gif_cma(name, max_steps=500):
    path = name + ".json"
    
    with open(path, "r") as f:
        solution = json.load(f)
    solution["robot"] = np.array(solution["robot"])
    solution["genes"] = np.array(solution["genes"])
    a = Agent(Network, solution, genes=solution["genes"])
    a.fitness = solution["fitness"]
    
    
    env = make_env(solution["env_name"], robot=solution["robot"], render_mode="rgb_array")
    env.metadata.update({'render_modes': ["rgb_array"]})
    
    a.fitness, imgs = evaluate(a, env, render=True, max_steps=max_steps)
    
    env.close()
    print("Fichier : ", name)   
    print(f"Morphology :")
    print(solution["robot"])
    print(f"Fitness avec {max_steps} steps : {a.fitness}")
    
    # Save the images as a gif
    imageio.mimsave(name + '.gif', imgs, duration=(1/50.0))


def best_walker():
    name = "project/evaluation/data/Walker/Walker62"
    create_gif_cma(name=name, max_steps=500)
    name = "project/evaluation/data/Walker/Walker63"
    create_gif_cma(name=name, max_steps=500)

def best_thrower(): 
    name = "project/evaluation/data/Thrower/BestThrower"
    create_gif_cma(name=name, max_steps=500)
    name = "project/evaluation/data/Thrower/ThrowerFinal"
    create_gif_cma(name=name, max_steps=500)

def best_climber():
    name = "project/evaluation/data/BestClimber"
    create_gif_cma(name=name, max_steps=500)

if __name__ == "__main__":
    #best_walker()
    best_thrower()
    #best_climber()  