import sys

# Add the "code library" folder to the Python path
sys.path.append('/mnt/data/code library')

# Import everything from the modules
from enkf import *
from bk_rsf_1d import *

import os
import numpy as np
import random

name_exp='enkf'

# Test for Burridge-Knopoff RSF 1D Ensemble Forward Model 
# 2023/09/27
# by Hamed Ali Diab-Montero
# h.a.diabmontero@tudelft.nl\

#----------------------------------
# TRUTH
#----------------------------------
N=20;  # Number of blocks
n_x=4; # Including theta, u, v

# parameters
eps=0.3
xi=0.5
gamma_lambda=np.sqrt(0.2)
gamma_mu=0.5
f=3.2

# Time step
tstep=1/100;
total_time=500;
nt=int(total_time/tstep)
warm_nt = int(200/tstep)                 # number of warm-up time steps

folder_truth='/data/bk_rsf_1d/BK_RSF_1D_test_datasets_periodic/truth/'

filename_data_theta_truth=os.path.join(folder_truth,'truth_theta_bk1d_periodic.txt')
filename_data_v_truth=os.path.join(folder_truth,'truth_v_bk1d_periodic.txt')
filename_data_u_truth=os.path.join(folder_truth,'truth_u_bk1d_periodic.txt')
filename_data_tau_truth=os.path.join(folder_truth,'truth_tau_bk1d_periodic.txt')

filename_time_truth=os.path.join(folder_truth,'time_truth_bk1d_periodic.txt')

theta_truth=np.genfromtxt(filename_data_theta_truth)
v_truth=np.genfromtxt(filename_data_v_truth)
u_truth=np.genfromtxt(filename_data_u_truth)
tau_truth=np.genfromtxt(filename_data_tau_truth)
t_truth=np.genfromtxt(filename_time_truth)

print('Finished loading the truth')

#---------------------------
# Observations
#---------------------------

obs_rate = 225     # interval of time steps between observations
obs_den = 2       # observation density (every obs_den -th grids are observed)
t_obs = t_truth[::obs_rate]
index_truth = range(len(t_truth))

t_first_da = int(warm_nt/obs_rate)+1 # index first assimilation 

folder_obsnet='/data/bk_rsf_1d/BK_RSF_1D_test_datasets_periodic/obsnet/'
filename_time_obsnet_c3=os.path.join(folder_obsnet,'obsnet_c3','obsnet_time_bk1d_periodic_obs_c3.txt')
filename_tau_obsnet_c3=os.path.join(folder_obsnet,'obsnet_c3','obsnet_tau_bk1d_periodic_obs_c3.txt')
filename_theta_obsnet_c3=os.path.join(folder_obsnet,'obsnet_c3','obsnet_theta_bk1d_periodic_obs_c3.txt')
filename_u_obsnet_c3=os.path.join(folder_obsnet,'obsnet_c3','obsnet_u_bk1d_periodic_obs_c3.txt')
filename_v_obsnet_c3=os.path.join(folder_obsnet,'obsnet_c3','obsnet_vel_bk1d_periodic_obs_c3.txt')

tau_rank_truth=np.transpose(tau_truth[::obs_rate,::obs_den])
theta_rank_truth=np.transpose(theta_truth[::obs_rate,::obs_den])
u_rank_truth=np.transpose(u_truth[::obs_rate,::obs_den])
v_rank_truth=np.transpose(v_truth[::obs_rate,::obs_den])

min_obs_rate=45

data_time_obsnet_c3=np.genfromtxt(filename_time_obsnet_c3)
time_obsnet=data_time_obsnet_c3[:]
t_obs=time_obsnet[::int(obs_rate/min_obs_rate)]

tau_obsnet = np.ones((len(t_obs), N)) * -999
theta_obsnet = np.ones((len(t_obs), N)) * -999
u_obsnet = np.ones((len(t_obs), N)) * -999
v_obsnet = np.ones((len(t_obs), N)) * -999

data_tau_obsnet_c3=np.genfromtxt(filename_tau_obsnet_c3)
tau_obsnet[:,::obs_den]=data_tau_obsnet_c3[::int(obs_rate/min_obs_rate),::obs_den]

data_theta_obsnet_c3=np.genfromtxt(filename_theta_obsnet_c3)
theta_obsnet[:,::obs_den]=data_theta_obsnet_c3[::int(obs_rate/min_obs_rate),::obs_den]

data_u_obsnet_c3=np.genfromtxt(filename_u_obsnet_c3)
u_obsnet[:,::obs_den]=data_u_obsnet_c3[::int(obs_rate/min_obs_rate),::obs_den]

data_v_obsnet_c3=np.genfromtxt(filename_v_obsnet_c3)
v_obsnet[:,::obs_den]=data_v_obsnet_c3[::int(obs_rate/min_obs_rate),::obs_den]

# R-matrix
n_obs = np.sum(v_obsnet[t_first_da, :] > -999)

r_eps_tau = np.sqrt(0.6)**2
r_eps_vel = np.sqrt(0.6)**2

R=np.eye(2*n_obs)
R[0:n_obs,0:n_obs]=R[0:n_obs,0:n_obs]*(r_eps_tau**2)
R[n_obs:2*n_obs,n_obs:2*n_obs]=R[n_obs:2*n_obs,n_obs:2*n_obs]*(r_eps_vel**2)
R_inv=np.linalg.inv(R)
print('Finished loading the observations')


#-----------------------------------------
# Ensemble Kalman FIlter
#-----------------------------------------

n_mem=100;                  # N - Ensemble size
                            # Bandwith hyperparameter

# ctlmean = X_truth[:, 0] + np.random.multivariate_normal(np.zeros(n_x)+1.5*np.ones(n_x), np.eye(n_x)).T

# initial condition
n_t=len(t_truth)
X_t = np.zeros((n_x*N, n_mem, n_t))
# Q = 2 * np.eye(n_x)            # background error covariance (only for the initial perturbation)
# Q_inv = np.linalg.inv(Q)
# X_t_2= np.random.multivariate_normal(ctlmean, Q, n_mem).T

folder_prior='/data/bk_rsf_1d/BK_RSF_1D_test_datasets_periodic/prior/'
filename_theta_prior=os.path.join(folder_prior,'prior_theta_bk1d_20_100_particles_periodic.txt')
filename_u_prior=os.path.join(folder_prior,'prior_u_bk1d_20_100_particles_periodic.txt')
filename_v_prior=os.path.join(folder_prior,'prior_v_bk1d_20_100_particles_periodic.txt')

# Replacing the values with the initial value of X_0
x_vector = np.zeros((n_x*N, 1))   # state vector

X_0_theta = np.loadtxt(filename_theta_prior)
X_0_u = np.loadtxt(filename_u_prior)
X_0_v = np.loadtxt(filename_v_prior)
X_0_tau= f + X_0_theta + np.log(X_0_v+1)

X_t[0:N,:,0]=np.transpose(X_0_theta) 
X_t[N:2*N,:,0]=np.transpose(X_0_u)
X_t[2*N:3*N,:,0]=np.transpose(X_0_v)
X_t[3*N:4*N,:,0]=np.transpose(X_0_tau)

#X_0 = np.loadtxt(filename_prior)
#X_t[:,:,0] = X_0.copy()

print('Finished loading the prior')

t_first_da = int(warm_nt/obs_rate)+1 # index first assimilation 
n_obs = np.sum(u_obsnet[t_first_da,:] > -999)

t = warm_nt
t_assim = len(t_truth)-obs_rate 

for k in range(warm_nt):
    X_t[0:N,:,k+1],X_t[N:2*N,:,k+1],X_t[2*N:3*N,:,k+1] = rk4_bk_1d_ensemble(N,n_mem,X_t[0:N,:,k],X_t[N:2*N,:,k],X_t[2*N:3*N,:,k],tstep)
    X_t[3*N:4*N,:,k+1]=f+X_t[0*N:1*N,:,k+1]+np.log(X_t[2*N:3*N,:,k+1]+1)

d = 0 # 1st observation

rank_histogram_tau=np.zeros(n_mem+1)
rank_histogram_v=np.zeros(n_mem+1)

list_diag_dfs=[]
list_iga=[]

list_assim=[]
list_sensitivity=[]

list_prior=[]
list_obs=[]
list_posterior=[]
list_preupdate=[]
list_update=[]

list_append_time_update=[]

while t*tstep < t_assim*tstep:
    
    t_analysis = int(t_obs[d]/tstep)

    # I need to start in k-1 beacause of the index system of python
    # We need to stop in t_analysis because of the index system of python
    for k in range(t+1, t_analysis+1,1):
        X_t[0:N,:,k],X_t[N:2*N,:,k],X_t[2*N:3*N,:,k] = rk4_bk_1d_ensemble(N,n_mem,X_t[0:N,:,k-1],X_t[N:2*N,:,k-1],X_t[2*N:3*N,:,k-1],tstep)
        X_t[3*N:4*N,:,k]=f+X_t[0*N:1*N,:,k]+np.log(X_t[2*N:3*N,:,k]+1)
   
    if(np.min(X_t[2*N:3*N,:,:])<=-1):
        print(f'Warning: A velocity below the threshold at d: {d} from forward modeling')

    t = t_analysis
        
    var_rank=10
    tau_ensemble=f+X_t[var_rank,:,t]+np.log(X_t[2*N+var_rank,:,t]+1)
    list_ensemble_tau= tau_ensemble.tolist() # To do the histogram on the prior
    list_ensemble_v=X_t[2*N+var_rank,:,t].tolist() # To do the histogram on the prior
    
    # Observations vectors
    y_t = np.ones((n_x*N, 1))*-999
    index_obs = np.where(v_obsnet[d, :] > -999)[0] # Experiment specific
    num_obs=len(index_obs)
    #y_t[:, 0] = x_obset[d, 2:] # Experiment specific
    #y_t[0:N, 0][index_obs] t_analysis = int(np.ceil(t_obs[d]/tstep))= theta_obsnet[d,:][index_obs] # Observations theta
    #y_t[N:2*N, 0][index_obs] = u_obsnet[d,:][index_obs] # Observations slip
    y_t[2*N:3*N, 0][index_obs] = v_obsnet[d,:][index_obs] # Observations slip-rate
    y_t[3*N:4*N, 0][index_obs] = tau_obsnet[d,:][index_obs] # Observations slip-rate

    # Prior ensemble for data assimilation
    X_da_prior=X_t[:, :, t].copy()
    X_da_prior[2*N:3*N,:]=np.log(X_da_prior[2*N:3*N, :]+1)

    list_prior.append(X_da_prior)
    list_preupdate.append(X_t[:, :, t].copy())
    # Ensemble Kalman Filter part of the method
    # We need to access t-1 because of the inx system of python
    #post_enkf = enkf(n_mem, n_x*N, X_t[:, :, t-1], y_t, R)
    post_enkf = enkf(n_mem, n_x*N, X_da_prior, y_t, R)

    X_ens=post_enkf['posterior']
    X_ens_mean=post_enkf['mean_post']
    P_ens=post_enkf['cov_post']
    
    # We need to access t-1 because of the inx system of python
    
    # Posterior ensemble for data assimilation
    X_da_posterior= X_ens.copy()
    
    # Adding prior, obs, posterior
    list_obs.append(y_t)
    list_posterior.append(post_enkf['posterior'])
    
    #print(X_da_posterior.shape)
    list_append_time_update.append(t)
    
   # Update theta
    X_t[0:N, :, t] = X_da_posterior[0:N, :]
    # Update u
    X_t[N:2*N, :, t] = X_da_posterior[N:2*N, :]
    # Update v
    X_t[2*N:3*N, :, t] = np.exp(X_da_posterior[2*N:3*N, :])-1   
    # Update tau
    X_t[3*N:4*N, :, t] = X_da_posterior[3*N:4*N, :]

    list_update.append(X_t[:, :, t].copy())
    
    #print(q_vector.shape)
    
    P_t = P_ens   
    
    if(np.min(X_t[2*N:3*N,:,:])<=-1):
        print(f'Warning: A velocity below the threshold at d: {d} from data assimilation')
    
        #Rank Histogram
    if(d<v_obsnet.shape[0]):
        #var_rank=10
        
        #list_ensemble=X_ens[var_rank,:].tolist()
        list_ensemble_v.sort()
        index_bin_v=np.where(list_ensemble_v>v_rank_truth[int(var_rank/obs_den),d])[0]
        if(1>len(index_bin_v)):
            rank_histogram_v[0]+=1
        elif(n_mem==len(index_bin_v)):
            rank_histogram_v[-1]+=1
        else:
            loc_bin_v=index_bin_v[0]+1
            rank_histogram_v[loc_bin_v-1]+=1
          
        list_ensemble_tau.sort()
        index_bin_tau=np.where(list_ensemble_tau>tau_rank_truth[int(var_rank/obs_den),d])[0]
        if(1>len(index_bin_tau)):
            rank_histogram_tau[0]+=1
        elif(n_mem==len(index_bin_tau)):
            rank_histogram_tau[-1]+=1
        else:
            loc_bin_tau=index_bin_tau[0]+1
            rank_histogram_tau[loc_bin_tau-1]+=1
            
        list_assim.append(t_obs[d])
            
    d += 1
    
    print(f'Finished with obs: {d}')
    
# samples_v=sample_histogram(n_mem, rank_histogram_v, n_samples=1000)
# samples_tau=sample_histogram(n_mem, rank_histogram_tau, n_samples=1000)
    
print('Finished with the estimation')



#--------------------------------
# RMSE ESTIMATION
#--------------------------------

#-------------------
#  THETA
#--------------------
index_variable=0
error_theta=theta_truth[:,:]-np.transpose(np.mean(X_t[index_variable*N:(index_variable+1)*N,:,:],axis=1))
error_theta_squared=error_theta**2
rmse_theta=np.sqrt((1/N)*np.sum(error_theta_squared,axis=1))

filename_estimate_theta='./results_rmse_test/rmse_theta_bk1d_periodic_obs_den_'+str(obs_den).zfill(2)+'_run_'+name_exp+'.txt'
np.savetxt(filename_estimate_theta,rmse_theta)

#--------------------------------
# SLIP
#--------------------------------
index_variable=1
error_u=u_truth[:,:]-np.transpose(np.mean(X_t[index_variable*N:(index_variable+1)*N,:,:],axis=1))
error_u_squared=error_u**2
rmse_u=np.sqrt((1/N)*np.sum(error_u_squared,axis=1))

filename_estimate_u='./results_rmse_test/rmse_u_bk1d_periodic_obs_den_'+str(obs_den).zfill(2)+'_run_'+name_exp+'.txt'
np.savetxt(filename_estimate_u,rmse_u)


#--------------------------------
# VELOCITY
#--------------------------------
index_variable=2
error_v=v_truth[:,:]-np.transpose(np.mean(X_t[index_variable*N:(index_variable+1)*N,:,:],axis=1))
error_v_squared=error_v**2
rmse_v=np.sqrt((1/N)*np.sum(error_v_squared,axis=1))

filename_estimate_v='./results_rmse_test/rmse_v_bk1d_periodic_obs_den_'+str(obs_den).zfill(2)+'_run_'+name_exp+'.txt'
np.savetxt(filename_estimate_v,rmse_v)


#--------------------------------
# TAU
#--------------------------------
index_variable=3
error_tau=tau_truth[:,:]-np.transpose(np.mean(X_t[index_variable*N:(index_variable+1)*N,:,:],axis=1))
error_tau_squared=error_tau**2
rmse_tau=np.sqrt((1/N)*np.sum(error_tau_squared,axis=1))

filename_estimate_tau='./results_rmse_test/rmse_tau_bk1d_periodic_obs_den_'+str(obs_den).zfill(2)+'_run_'+name_exp+'.txt'
np.savetxt(filename_estimate_tau,rmse_tau)





