import numpy as np

def h_operator(nx, obs_vect):
    """
    Creates an observation operator matrix for mapping state space to observation space.

    Parameters:
    - nx (int): The dimension of the state space.
    - obs_vect (numpy.ndarray): The observation vector, containing observations of the system.

    Returns:
    - numpy.ndarray: The observation operator matrix.

    Description:
    The function generates a matrix that maps the state space (of dimension nx) to the observation space. It is used in filtering algorithms to relate the state estimates to the observations. The function performs the following steps:
    1. Identifies valid observations in the observation vector (ignoring entries marked with -999).
    2. Initializes a zero matrix of size (num_obs, nx), where num_obs is the number of valid observations.
    3. Sets the corresponding entries in the matrix to 1 based on the indices of the valid observations.

    Note:
    - The observation vector is assumed to contain invalid observations marked with -999, which are filtered out.
    - The function assumes that the indices in the observation vector correspond to the state variables' indices in the state space.
    - The function uses numpy for matrix operations.
    """
    
    # Identify indices of valid observations
    index_obs = np.where(obs_vect > -999)[0]
    num_obs = len(index_obs)
    
    # Initialize the observation operator matrix
    h_matrix = np.zeros((num_obs, nx))
    
    # Set the corresponding entries in the matrix to 1
    for i in range(num_obs):
        h_matrix[i, index_obs[i]] = 1
        
    return h_matrix

