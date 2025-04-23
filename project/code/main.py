from evolution import *


if __name__ == "__main__":
    
    config = {
    "env_name": "Walker-v0",
    "robot": walker,
    "generations": 10, # To change: increase!
    "lambda": 10,
    "max_steps": 100, # to change to 500
    }

    a = one_plus_lambda(config)
    # a.fitness
    env = make_env(config["env_name"], robot=config["robot"])
    reward = evaluate(a, env, render=False)
    print(f"Reward: {reward}")
    env.close()
   
    np.save("Walker0.npy", a.genes)