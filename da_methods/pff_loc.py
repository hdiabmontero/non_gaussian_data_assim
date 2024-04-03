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


def grad_log_post(H, R, R_inv, y, y_i, B, x_s_i, x0_mean):
    """
    Calculate the gradient of the log posterior distribution.

    Args:
    H (numpy.array): Observation operator matrix.
    R (numpy.array): Observation error covariance matrix.
    R_inv (numpy.array): Inverse of R.
    y (numpy.array): Observation vector.
    y_i (numpy.array): Individual observation vector for a particle.
    B (numpy.array): Covariance matrix of the ensemble.
    x_s_i (numpy.array): Current state of a particle.
    x0_mean (numpy.array): Mean state of the prior distribution.

    Returns:
    numpy.array: Gradient of the log posterior.
    """
    obs_part = B.dot(H.transpose()).dot(R_inv).dot(y - y_i)[:, 0]
    prior_part = x_s_i - x0_mean
    grad_log_post_est = obs_part - prior_part

    return grad_log_post_est

def pff(n_mem, n_states, ensemble, obs_vect, index_obs, N, r_influ):
    """
    Implement the Particle Flow Filter.

    Args:
    n_mem (int): Number of ensemble members.
    n_states (int): Number of states.
    ensemble (numpy.array): Initial ensemble of states.
    obs_vect (numpy.array): Observation vector.
    index_obs (numpy.array): Indices of valid observations.
    N (int): The number of grid points.
    r_influ (float): Radius of influence for localization.

    Returns:
    dict: Dictionary containing the posterior ensemble, mean, and covariance.
    """
    B = np.cov(ensemble)
    
    # Apply localization to the prior covariance matrix
    B=localization(r_influ,N,B)
    
    x0_mean = np.mean(ensemble, axis=1)

    # Pseudo-time flow parameters
    s = 0
    max_s = 100
    ds = 0.05 / 10
    alpha = 0.05 / 10  # Tuning parameter for the covariance of the kernel

    x_s = ensemble.copy()
    python_pseudoflow = np.zeros((n_states, n_mem, max_s + 1))
    python_pseudoflow[:, :, 0] = x_s.copy()

    n_obs = np.sum(obs_vect > -999)

    # Pseudo-time for data assimilation
    while s < max_s:
        print(f'Iteration: {s}')

        H = np.zeros((n_obs, n_states))
        Hx = np.zeros((n_obs, n_mem))
        dHdx = np.zeros((n_obs, n_states, n_mem))

        for i in range(n_mem):
            H = h_operator(n_states, obs_vect)
            Hx[:, :] = x_s[index_obs, :]
            y = np.ones((n_obs, 1))
            y[:, 0] = obs_vect[index_obs]
            y_i = np.ones((n_obs, 1))
            y_i[:, 0] = Hx[:, i]

            dpdx[:, i] = grad_log_post(H, R, R_inv, y, y_i, B, x_s[:, i], x0_mean)

        # Kernel calculation
        B_d = np.zeros((n_states))
        for d in range(n_states):
            B_d[d] = B[d, d]

        kernel = np.zeros((n_states, n_mem, n_mem))
        dkdx = np.zeros((n_states, n_mem, n_mem))
        I_f = np.zeros((n_states, n_mem))

        for i in range(n_mem):
            for j in range(i, n_mem):
                kernel[:, i, j] = np.exp((-1 / 2) * ((x_s[:, i] - x_s[:, j]) ** 2) / (alpha * B_d[:]))
                dkdx[:, i, j] = ((x_s[:, i] - x_s[:, j]) / alpha) * kernel[:, i, j]
                if j != i:
                kernel[:, i, j] = kernel[:, j, i]
                dkdx[:, i, j] = -dkdx[:, j, i]

            attractive_term = (1 / n_mem) * (kernel[:, i, :] * dpdx)
            repelling_term = (1 / n_mem) * dkdx[:, i, :]
            I_f[:, i] = np.sum(attractive_term + repelling_term, axis=1)

        # Update the state vector for next pseudo time step
        fs = I_f
        x_s += ds * fs
        python_pseudoflow[:, :, s + 1] = x_s

        s += 1

    # Gathering final results
    posterior_vect = python_pseudoflow[:, :, -1]
    mean_posterior = np.mean(posterior_vect, axis=1)
    cov_posterior = np.cov(posterior_vect)
            
    pff_pseudoflow = {"posterior": posterior_vect, "mean_post": mean_posterior, "cov_post": cov_posterior}
        
    return pff_pseudoflow

