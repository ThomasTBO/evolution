
from environnement import *
from imports import *
from scipy.ndimage import label

def is_connected(matrix):
    """
    Vérifie si la matrice est connectée (toutes les cases non nulles forment un seul composant connexe).
    """
    labeled, num_features = label(matrix != 0)  # Label les composants connexes
    return num_features == 1

    
    
def mutate(morphology, probability=1/25):
    """
    Mutate the morphology matrix (5x5) based on the given rules:
    - If the square is non-zero, mutate it to a value between 1 and 5 with a probability of 1/25.
    - If the square is zero and has a non-zero adjacent square, mutate it to a value between 0 and 5 with a probability of 1/25.
    """
    rows, cols = morphology.shape
    new_morphology = morphology.copy()
     #PROBABILITY 1/n

    for i in range(rows):
        for j in range(cols):
            if np.random.rand() < probability: 
                if morphology[i][j] != 0:
                    # Mutate non-zero squares
                    original_value = new_morphology[i][j]
                    new_square= np.random.randint(0, 5)  # Peut inclure 0
                    new_morphology[i][j] = new_square
                    if (new_square == 0) and (not is_connected(new_morphology)) :
                        new_morphology[i][j] = original_value  # Annuler la mutation si elle déconnecte
                    
                else:
                    # Check for non-zero adjacent squares
                    has_non_zero_adjacent = False
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Adjacent directions
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols and morphology[ni][nj] != 0:
                            has_non_zero_adjacent = True
                        break
                    # Mutate empty squares with non-zero adjacent squares 
                    if has_non_zero_adjacent :
                        new_morphology[i][j] = np.random.randint(0, 5)

    return new_morphology

def mutate_climber(morphology, probability=1/25):
    """
    Mutate the morphology matrix (5x5) based on the given rules:
    - If the square is non-zero, mutate it to a value between 1 and 5 with a probability of 1/25.
    - If the square is zero and has a non-zero adjacent square, mutate it to a value between 0 and 5 with a probability of 1/25.
    """
    rows, cols = morphology.shape
    new_morphology = morphology.copy()
     #PROBABILITY 1/n

    for i in range(rows):
        for j in range(cols):
            if np.random.rand() < probability: 
                if morphology[i][j] != 0:
                    # Mutate non-zero squares
                    original_value = new_morphology[i][j]
                    if i == 4 : 
                        new_square= np.random.randint(0, 4)  
                    else : 
                        new_square= np.random.randint(0, 5)  # Peut inclure 0
                    
                    new_morphology[i][j] = new_square

                    if (new_square == 0) and (not is_connected(new_morphology)) :
                        new_morphology[i][j] = original_value  # Annuler la mutation si elle déconnecte
                    
                else:
                    # Check for non-zero adjacent squares
                    has_non_zero_adjacent = False
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Adjacent directions
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols and morphology[ni][nj] != 0:
                            has_non_zero_adjacent = True
                        break
                    # Mutate empty squares with non-zero adjacent squares 
                    if has_non_zero_adjacent :
                        if i == 4 : 
                            new_square= np.random.randint(0, 4)  
                        else : 
                            new_square= np.random.randint(0, 5)  

                        new_morphology[i][j] = new_square

    return new_morphology

def mutate_climber_2(morphology, probability=1/25):
    """
    Mutate the morphology matrix (5x5) based on the given rules:
    - If the square is non-zero, mutate it to a value between 1 and 5 with a probability of 1/25.
    - If the square is zero and has a non-zero adjacent square, mutate it to a value between 0 and 5 with a probability of 1/25.
    """
    rows, cols = morphology.shape
    new_morphology = morphology.copy()
     #PROBABILITY 1/n

    for i in range(rows):
        for j in range(cols):
            if np.random.rand() < probability: 
                if morphology[i][j] == 1 :
                    new_morphology[i][j] = 2 
                elif morphology[i][j] == 2 :
                    new_morphology[i][j] = 1
                elif morphology[i][j] == 3 :
                    new_morphology[i][j] = 4
                elif morphology[i][j] == 4 :
                    new_morphology[i][j] = 3    
    return new_morphology

# morphology2 = np.array([
#     [3, 3, 3, 3, 3],
#     [3, 3, 3, 3, 3],
#     [3, 3, 3, 3, 3],
#     [3, 3, 3, 3, 3],
#     [3, 3, 3, 3, 3]
# ])
# morphology = np.array([
#     [1, 1, 3, 1, 1],
#     [1, 1, 0, 1, 1],
#     [1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1],
# ])

# new_morphology = mutate_climber_2(morphology, 25/25)
# print("Original Morphology:")
# print(morphology)
# print("Mutated Morphology:")
# print(new_morphology)   

# config = {
#     "env_name": "Climber-v2",
#     "robot": morphology,
#     "generations": 10, # To change: increase!
#     "lambda": 10,
#     "max_steps": 100, # to change to 500
#     }
# from evosim_ray import EvoGymEnv
# from environnement import *
# from agents import *

# cfg = get_cfg(config["env_name"], robot=config["robot"]) # Get network dims
# cfg = {**config, **cfg} # Merge configs
# # env = EvoGymEnv(cfg["env_name"], robot=cfg["robot"])


# ex_agent = Agent(Network, cfg)
# genes = ex_agent.genes
# print(genes.shape)

# config2 = {
#     "env_name": "Climber-v2",
#     "robot": new_morphology,
#     "generations": 10, # To change: increase!
#     "lambda": 10,
#     "max_steps": 100, # to change to 500
#     }
# cfg2 = get_cfg(config2["env_name"], robot=config2["robot"]) # Get network dims
# cfg2 = {**config2, **cfg2} # Merge configs
# # env = EvoGymEnv(cfg2["env_name"], robot=cfg2["robot"])
# ex_agent2 = Agent(Network, cfg2)
# genes2 = ex_agent2.genes
# print(genes2.shape)
