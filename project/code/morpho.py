
from environnement import *
import cma
import json
from gif import create_gif
from imports import *
from torch import sigmoid
from gif_cma import create_gif_cma


    
    
def mutate_morphology(morphology, probability=1/25):
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
            if morphology[i][j] != 0:
                # Mutate non-zero squares with probability 1/25
                if np.random.rand() < probability:
                    new_morphology[i][j] = np.random.randint(1, 6)
            else:
                # Check for non-zero adjacent squares
                has_non_zero_adjacent = False
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Adjacent directions
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and morphology[ni][nj] != 0:
                        has_non_zero_adjacent = True
                        break
                
                # Mutate empty squares with non-zero adjacent squares with probability 1/25
                if has_non_zero_adjacent and np.random.rand() < probability:
                    new_morphology[i][j] = np.random.randint(0, 6)

    return new_morphology


