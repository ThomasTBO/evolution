from evosim_ray import *
from agents import *
from environnement import *


# @ray.remote
# def evaluate_agent(cfg,genes,env):
#     #env = EvoGymEnv(cfg["env_name"], cfg["robot"])
#     agent = Agent(Network, cfg, genes=genes)
#     fitness = evaluate(agent, env, max_steps=cfg["max_steps"])
#     env.close()
#     return - fitness

@ray.remote
def evaluate_agent(env, genes, cfg):
    """
    Evaluate the environment for a given number of steps.
    """
    agent = Agent(Network, cfg, genes=genes)
    obs = env.reset()
    done = False
    value = 0
    print("TEST")
    for _ in range(cfg["max_steps"] ):
        # action = env.action_space.sample()
        action = agent.act(obs)  # Use the agent to get the action
        obs, reward, done, trunc,  info = env.step(action)
        value += reward
    return value
    