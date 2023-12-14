import numpy as np
import h_operator

def enkf(mem,nx,ensemble,obs_vect,R):
    
    """
    Implements the Ensemble Kalman Filter (EnKF) algorithm.

    Parameters:
    mem (int): The number of ensemble members.
    nx (int): The dimension of the state space.
    ensemble (numpy.ndarray): The ensemble matrix, representing the state estimates. Each column is an ensemble member.
    obs_vect (numpy.ndarray): The observation vector, containing observations of the system.
    R (numpy.ndarray): The observation error covariance matrix.

    Returns:
    dict: A dictionary containing the following key-value pairs:
        - 'posterior' (numpy.ndarray): The posterior ensemble matrix after assimilating observations.
        - 'kalman_gain' (numpy.ndarray): The Kalman gain matrix.
        - 'innovation' (numpy.ndarray): The innovation vector, representing the difference between actual and predicted observations.
        - 'mean_post' (numpy.ndarray): The mean of the posterior ensemble.
        - 'cov_post' (numpy.ndarray): The covariance matrix of the posterior ensemble.

    Description:
    The function performs the following steps:
    1. Filters the observation vector to remove invalid observations.
    2. Organizes the prior ensemble matrix and calculates its mean and covariance.
    3. Perturbs the observation vector for each ensemble member.
    4. Calculates the observation operator matrix (h_matrix) using an external function 'h_operator'.
    5. Computes the Kalman gain, which blends the uncertainty in the prior with the uncertainty in the observation.
    6. Calculates the innovation, i.e., the discrepancy between observed and predicted observations.
    7. Updates the ensemble (posterior) using the Kalman gain and the innovation.
    8. Computes the mean and covariance of the posterior ensemble.
    9. Returns a dictionary with the results.
    
    Note:
    - The function assumes the availability of an external function 'h_operator' for computing the observation operator.
    - Observations marked with -999 are considered invalid and filtered out.
    - The function uses numpy for matrix operations.
    """
    
    index_obs=np.where(obs_vect>-999)[0]
    num_obs=len(index_obs)

    # Organize the prior vector
    #prior_vect=np.transpose(ensemble)
    prior_vect=ensemble
    
    # Calculate the prior covariance matrix
    mean_prior=np.mean(prior_vect,axis=1)
    cov_prior=np.cov(prior_vect)
    
#     # Perturbed observation vector
    obs_vect_filtered=obs_vect[index_obs]
    obs_vect_perturbed=np.zeros((num_obs,mem))
    r_obs_vect=np.diag(R)
    
    for i in range(mem):
        for j in range(num_obs):
            obs_vect_perturbed[j,i]=obs_vect_filtered[j,0]
  
    #cov_obs=np.identity(num_obs)*r_obs**2
    cov_obs=R[:,:]
    
#     # Calculate the observation operator
    h_matrix=h_operator(nx,obs_vect)  
    
    #print(h_matrix.shape)
    
#     # Calculate the kalman gain
    
    k_left=cov_prior.dot(np.transpose(h_matrix))
    k_right=h_matrix.dot(cov_prior).dot(np.transpose(h_matrix))+cov_obs
    k_right_inv=np.linalg.inv(k_right)

    kalman_gain=k_left.dot(k_right_inv)
    
#     # Calculate the innovation
    
    innovation=obs_vect_perturbed-h_matrix.dot(prior_vect)

    
#     # Calculate the posterior
    
    posterior_vect=prior_vect+kalman_gain.dot(innovation)

    mean_posterior=np.mean(posterior_vect, axis=1)
    
    cov_posterior=np.cov(posterior_vect)
    
#     res_matrix=np.transpose(h_matrix).dot(k_right_inv).dot(h_matrix).dot(cov_prior)
    
    enkf={"posterior":posterior_vect,"kalman_gain":kalman_gain,"innovation":innovation,
          "mean_post":mean_posterior,"cov_post":cov_posterior}
    
    return enkf
