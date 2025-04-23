import ray
from evolution import *
from imports import *
from environnement import *

walker = np.array([
    [3, 3, 3, 3, 3],
    [3, 3, 3, 0, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3],
    [3, 3, 0, 3, 3]
    ])

class EvoSim:
    def __init__(self, env_name, robot=None):
        if robot is None: 
            self.env = gym.make(env_name)
        else:
            self.env = gym.make(env_name, body=robot)
        self.env_name = env_name
        self.robot = robot

    def __reduce__(self):
        deserializer = EvoSim
        serialized_data = (self.env_name,self.robot, )
        return deserializer, serialized_data


config = {
    "env_name": "Walker-v0",
    "robot": walker,
    "generations": 10, # To change: increase!
    "lambda": 10,
    "max_steps": 100, # to change to 500
    }

ray.init()    
   
env = EvoSim(config)

original = env
# print(original)

copied = ray.get(ray.put(original))
# # print(copied.conn)