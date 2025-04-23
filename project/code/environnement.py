from agents import *

walker = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 0, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3]
    ])

def make_env(env_name, seed=None, robot=None, **kwargs):
    if robot is None: 
        env = gym.make(env_name, **kwargs)
    else:
        connections = get_full_connectivity(robot)
        env = gym.make(env_name, body=robot, connections=connections, **kwargs)
    env.robot = robot
    if seed is not None:
        env.seed(seed)
        
    return env

def evaluate(agent, env, max_steps=500, render=False):
    obs, i = env.reset()
    agent.model.reset()
    reward = 0
    steps = 0
    done = False
    if render:
        imgs = []
    while not done and steps < max_steps:
        if render:
            img = env.render() #mode='img'
            imgs.append(img)
        action = agent.act(obs)
        obs, r, done, trunc,  _ = env.step(action)
        reward += r
        steps += 1
        
    if render:
        return reward, imgs
    return reward

def get_cfg(env_name, robot=None):
    env = make_env(env_name, robot=robot)
    cfg = {
        "n_in": env.observation_space.shape[0],
        "h_size": 32,
        "n_out": env.action_space.shape[0],
    }
    env.close()
    return cfg

def trucs():
    env_name = 'Thrower-v0'
    robot = walker
    env = make_env(env_name, robot=robot)
    cfg = get_cfg(env_name, robot=robot)
    a = Agent(Network, cfg)

    # Evaluation
    reward = evaluate(a, env, render=True)
    print(f"Reward: {reward}")
    env.close()
    env_name = 'Thrower-v0'
    robot = walker

    cfg = get_cfg(env_name, robot)
    a = Agent(Network, cfg)

    env = make_env(env_name, robot=walker)
    s = env.reset()
    print(len(s), s)

    # Evaluation
    env = make_env(env_name, robot=walker)
    reward = evaluate(a, env, render=True)
    print(f"Reward: {reward}")
    env.close()


def mp_eval(a, cfg):
    trucs()
    env = make_env(cfg["env_name"], robot=cfg["robot"])
    fit = evaluate(a, env, max_steps=cfg["max_steps"])
    env.close()
    return fit