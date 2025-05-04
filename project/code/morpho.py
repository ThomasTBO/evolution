
from environnement import *
import cma
import json
from gif import create_gif
from imports import *
from torch import sigmoid
from gif_cma import create_gif_cma
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
                    if has_non_zero_adjacent and np.random.rand() < probability:
                        new_morphology[i][j] = np.random.randint(0, 5)

    return new_morphology


# morphology = np.array([
#     [3, 3, 5, 0, 0],
#     [3, 3, 3, 0, 0],
#     [0, 3, 2, 0, 0],
#     [0, 3, 5, 0, 0],
#     [0, 3, 1, 0, 0]
# ])

# new_morphology = mutate(morphology, 1)
# print("Original Morphology:")
# print(morphology)
# print("Mutated Morphology:")
# print(new_morphology)        

    
