import numpy as np

def h_operator(nx, obs_vect):
    """
    Create the observation operator matrix H.
    Args:
    nx (int): The size of the state vector.
    obs_vect (numpy.array): Observation vector, where -999 indicates missing data.
    Returns:
    numpy.array: The observation operator matrix.
    """
    # Identify valid observations (not marked as -999) and create H matrix
    index_obs = np.where(obs_vect > -999)[0]
    h_matrix = np.zeros((len(index_obs), nx))
    for i in range(len(index_obs)):
        h_matrix[i, index_obs[i]] = 1
    return h_matrix

def localization(r_influ, N, cov_prior):
    """
    Apply localization to the covariance matrix using Gaussian-like decay.
    Args:
    r_influ (float): Radius of influence for localization.
    N (int): The number of grid points.
    cov_prior (numpy.array): Prior covariance matrix.
    Returns:
    numpy.array: Localized covariance matrix.
    """
    # Create a mask for localization based on Gaussian decay
    tmp = np.zeros((N, N))
    for i in range(1, 3 * r_influ + 1):
        tmp += np.exp(-i**2 / r_influ**2) * (np.diag(np.ones(N - i), i) + np.diag(np.ones(N - i), -i))
    mask = tmp + np.diag(np.ones(N))

    # Apply the mask to the prior covariance matrix
    cov_prior_loc = np.zeros(cov_prior.shape)
    for i in range(1, 4):
        for j in range(1, 4):
            block = (i - 1) * N, i * N, (j - 1) * N, j * N
            cov_prior_loc[block[0]:block[1], block[2]:block[3]] = np.multiply(cov_prior[block[0]:block[1], block[2]:block[3]], mask)
    return cov_prior_loc

def enkf(mem, nx, ensemble, obs_vect, R, N, r_influ):
    """
    Implement the Ensemble Kalman Filter with localization.
    Args:
    mem (int): Number of ensemble members.
    nx (int): Size of the state vector.
    ensemble (numpy.array): Ensemble of state estimates.
    obs_vect (numpy.array): Observation vector.
    R (numpy.array): Observation error covariance matrix.
    N (int): The number of grid points.
    r_influ (float): Radius of influence for localization.
    Returns:
    dict: Posterior ensemble, Kalman gain, innovation, mean and covariance of the posterior.
    """
    # Extract indices of valid observations and set up the prior state vector
    index_obs = np.where(obs_vect > -999)[0]
    prior_vect = ensemble

    # Compute mean and covariance of the prior, and apply localization
    mean_prior = np.mean(prior_vect, axis=1)
    cov_prior = np.cov(prior_vect)
    cov_prior = localization(r_influ, N, cov_prior)
    
    # Perturb the observation vector
    obs_vect_filtered = obs_vect[index_obs]
    obs_vect_perturbed = np.zeros((len(index_obs), mem))
    for i in range(mem):
        obs_vect_perturbed[:, i] = obs_vect_filtered

    # Set up the observation error covariance matrix and compute H matrix
    cov_obs = R[:, :]
    h_matrix = h_operator(nx, obs_vect)

    # Calculate the Kalman gain
    k_left = cov_prior.dot(h_matrix.T)
    k_right_inv = np.linalg.inv(h_matrix.dot(cov_prior).dot(h_matrix.T) + cov_obs)
    kalman_gain = k_left.dot(k_right_inv)

    # Calculate the innovation and update the posterior ensemble
    innovation = obs_vect_perturbed - h_matrix.dot(prior_vect)
    posterior_vect = prior_vect + kalman_gain.dot(innovation)

    # Compute mean and covariance of the posterior
    mean_posterior = np.mean(posterior_vect, axis=1)
    cov_posterior = np.cov(posterior_vect)

    # Output dictionary
    enkf_output = {
        "posterior": posterior_vect,
        "kalman_gain": kalman_gain,
        "innovation": innovation,
        "mean_post": mean_posterior,
        "cov_post": cov_posterior
    }
    return enkf_output
