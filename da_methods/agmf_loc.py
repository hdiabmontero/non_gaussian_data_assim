import numpy as np

def h_operator(nx, obs_vect):
    """
    Create the observation operator matrix H.
    
    Args:
    nx (int): Size of the state vector.
    obs_vect (numpy.array): Observation vector, where -999 indicates missing data.
    
    Returns:
    numpy.array: The observation operator matrix.
    """
    # Identifying indices of valid observations (not -999)
    index_obs = np.where(obs_vect > -999)[0]
    num_obs = len(index_obs)

    # Initializing the H matrix with zeros
    h_matrix = np.zeros((num_obs, nx))

    # Setting 1 at positions corresponding to actual observations
    for i in range(num_obs):
        h_matrix[i, index_obs[i]] = 1

    return h_matrix

def localization(r_influ,N,cov_prior):
    """
    Apply localization to the covariance matrix.
    
    Args:
    r_influ (float): The radius of influence for localization.
    N (int): The number of grid points.
    cov_prior (numpy.array): The prior covariance matrix.
    
    Returns:
    numpy.array: Localized covariance matrix.
    """
    # Create a localization mask with Gaussian-like decay
    tmp = np.zeros((N, N))
    for i in range(1, 3 * r_influ + 1):
        tmp += np.exp(-i**2 / r_influ**2) * (np.diag(np.ones(N - i), i) + np.diag(np.ones(N - i), -i))
    mask = tmp + np.diag(np.ones(N))

    # Apply the localization mask to the prior covariance matrix
    cov_prior_loc = np.zeros(cov_prior.shape)
    for i in range(1, 4):
        for j in range(1, 4):
            cov_prior_loc[(i - 1) * N:i * N, (j - 1) * N:j * N] = np.multiply(cov_prior[(i - 1) * N:i * N, (j - 1) * N:j * N], mask)

    return cov_prior_loc

def gaussian_mixt(weight_vect, n_obs, ens_vect, obs_vect, h_matrix, cov_matrix):
    """
    Compute the weights for ensemble members using Gaussian Mixture Model.

    Args:
    weight_vect (numpy.array): Current weights of the ensemble members.
    n_obs (int): Number of observations.
    ens_vect (numpy.array): Ensemble matrix.
    obs_vect (numpy.array): Observation vector.
    h_matrix (numpy.array): Observation operator matrix.
    cov_matrix (numpy.array): Covariance matrix of the observations.

    Returns:
    numpy.array: Updated weights for the ensemble members.
    """
    # Normalizing factor for Gaussian probability density function
    norm_factor = 1 / np.sqrt(((2 * np.pi) ** n_obs) * np.linalg.det(cov_matrix))
    weight_mixt = np.zeros(len(weight_vect))
    prob_dens = np.zeros(len(weight_vect))

    # Calculating the weights based on the Gaussian distribution
    for i in range(ens_vect.shape[1]):
        innovation = obs_vect[:, 0] - h_matrix.dot(ens_vect[:, i])
        prob_dens[i] = norm_factor * np.exp(-(1 / 2) * ((np.transpose(innovation)).dot(np.linalg.inv(cov_matrix).dot(innovation))))
        weight_mixt[i] = prob_dens[i] * weight_vect[i]

    # Normalizing the weights
    weight_final = weight_mixt / np.sum(weight_mixt)

    return weight_final

def randsample(n, p):
    """
    Perform resampling based on given probabilities.

    Args:
    n (int): Number of items to sample.
    p (numpy.array): Array of probabilities associated with each item.

    Returns:
    numpy.array: Array of indices, sampled according to probabilities p.
    """
    return np.random.choice(np.arange(0, n, 1), size=n, replace=True, p=p)

def agmf(mem, nx, ensemble, obs_vect, R, h, w_prev, nc_threshold,N, r_influ):
    """
    Implement the Adaptive Gaussian Mixture Filter.

    Args:
    mem (int): Number of ensemble members.
    nx (int): Size of the state vector.
    ensemble (numpy.array): Ensemble of state estimates.
    obs_vect (numpy.array): Observation vector.
    R (numpy.array): Observation error covariance matrix.
    h (float): Scale factor for covariance matrix.
    w_prev (numpy.array): Previous weights of the ensemble members.
    N (int): The number of grid points.
    nc_threshold (float): Threshold for deciding whether resampling is necessary.
    r_influ (float): Radius of influence for localization.

    Returns:
    dict: Dictionary containing the posterior ensemble, Kalman gain, innovation, 
          mean and covariance of the posterior, weights, and alpha value.
    """
    # Identifying indices of valid observations
    index_obs = np.where(obs_vect > -999)[0]
    num_obs = len(index_obs)

    # Preparing the prior state vector (ensemble matrix)
    prior_vect = ensemble

    # Calculating the mean and covariance of the prior
    mean_prior = np.mean(prior_vect, axis=1)
    cov_prior = (h ** 2) * np.cov(prior_vect)
    
    # Apply localization to the prior covariance matrix
    cov_prior=localization(r_influ,N,cov_prior)
    
    # Filtering and perturbing the observation vector
    obs_vect_filtered = obs_vect[index_obs]
    obs_vect_perturbed = np.zeros((num_obs, mem))
    for i in range(mem):
        for j in range(num_obs):
            obs_vect_perturbed[j, i] = obs_vect_filtered[j, 0]

    # Observation error covariance matrix
    cov_obs = R[:, :]

    # Calculating the observation operator matrix
    h_matrix = h_operator(nx, obs_vect)

    # Calculating the Kalman gain
    k_left = cov_prior.dot(np.transpose(h_matrix))
    k_right_inv = np.linalg.inv(h_matrix.dot(cov_prior).dot(np.transpose(h_matrix)) + cov_obs)
    kalman_gain = k_left.dot(k_right_inv)

    # Calculating the innovation
    innovation = obs_vect_perturbed - h_matrix.dot(prior_vect)

    # Calculating the posterior ensemble
    posterior_vect = prior_vect + kalman_gain.dot(innovation)
    mean_posterior = np.mean(posterior_vect, axis=1)
    cov_posterior = np.cov(posterior_vect)

    # Recalculating weights
    w_t = gaussian_mixt(w_prev, num_obs, posterior_vect, obs_vect_perturbed, h_matrix, R)

    # Evaluating degeneracy and calculating the bridging alpha
    N_eff = 1 / np.sum(w_t ** 2)
    alpha = N_eff / mem

    # Adjusting weights
    w_t = w_t * alpha + (1 - alpha) * (1 / mem)

    # Resampling if necessary
    resamp = 0
    if N_eff < nc_threshold:  
        J = randsample(mem, w_t)
        epsc = np.random.normal(0, 0.1, mem)
        for i in range(mem):
            posterior_vect[:, i] = posterior_vect[:, int(J[i])] + np.sqrt(np.diag(cov_posterior)) * epsc[i] 
        cov_posterior = (h ** 2) * np.cov(posterior_vect)
        resamp = 1

    # Result output
    agmf_output = {
        "posterior": posterior_vect,
        "kalman_gain": kalman_gain,
        "innovation": innovation,
        "mean_post": mean_posterior,
        "cov_post": cov_posterior,
        "weights": w_t,
        "alpha": alpha,
    }

    return agmf_output
