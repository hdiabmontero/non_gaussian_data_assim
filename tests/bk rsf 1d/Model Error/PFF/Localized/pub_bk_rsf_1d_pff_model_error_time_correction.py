import numpy as np
import random
import os

#----------------------------------------------
#   FUNCTIONS
#----------------------------------------------

def h_operator(nx,obs_vect):
    
    #nx=len(obs_vect)
    index_obs=np.where(obs_vect>-999)[0]
    num_obs=len(index_obs)
    
    h_matrix=np.zeros((num_obs,nx))
    for i in range(num_obs):
        h_matrix[i,index_obs[i]]=1
        
    return h_matrix

def sample_histogram(n_mem, rank_histogram, n_samples):
    bin_edges = np.arange(n_mem + 2)
    frequencies = np.array(rank_histogram)

    probabilities = frequencies / frequencies.sum()
    cumulative_probabilities = np.cumsum(probabilities)

    random_numbers = np.random.random(n_samples)
    samples = np.array([bin_edges[np.argwhere(cumulative_probabilities >= random_number)[0][0]] for random_number in random_numbers])
    
    return samples

def localization(r_influ,N,cov_prior):
    
    tmp = np.zeros((N, N))
    for i in range(1, 3*r_influ+1):
        tmp += np.exp(-i**2 / r_influ**2)*(np.diag(np.ones(N-i), i) + np.diag(np.ones(N-i), -i))
    mask = tmp + np.diag(np.ones(N))

    cov_prior_loc=np.zeros(cov_prior.shape)
    for i in range(1,3+1,1):
        for j in range(1,3+1,1):
            cov_prior_loc[(i-1)*N:i*N,(j-1)*N:j*N]=np.multiply(cov_prior[(i-1)*N:i*N,(j-1)*N:j*N],mask) # Multiplication step with the mask

    return cov_prior_loc
    


   
def rk4_bk_1d_step(N,theta_in,u_in,v_in,dt):
    
    # parameters
    eps=0.3
    xi=0.5
    gamma_lambda=np.sqrt(0.2)
    gamma_mu=0.5
    f=3.2
    
    #Initial conditions
    
    theta_00=theta_in[:]
    v_00=v_in[:]
    u_00=u_in[:]

    u_p1=np.zeros(N)
    u_p1[0:-1]=u_00[1:]
    u_p1[-1]=u_00[0]

    u_n1=np.zeros(N)
    u_n1[0]=u_00[1]
    u_n1[1:]=u_00[0:-1]

    #------------
    # K1
    #------------
    dthetadt=-(v_00+1)*(theta_00+(1+eps)*np.log(v_00+1))
    dudt=v_00
    dvdt= (gamma_mu**2)*(u_n1-2*u_00+u_p1) \
                    -(gamma_lambda**2)*u_00 -((gamma_mu**2)/(xi))*(f+theta_00+np.log(v_00+1))

    k11=dthetadt[:];
    k12=dudt[:];
    k13=dvdt[:];

    theta_00=theta_00+0.5*k11*dt;
    u_00=u_00+0.5*k12*dt;
    v_00=v_00+0.5*k13*dt;

    #------------
    # K2
    #------------
    u_p1[0:-1]=u_00[1:]
    u_p1[-1]=u_00[0]

    u_n1[0]=u_00[1]
    u_n1[1:]=u_00[0:-1]

    dthetadt=-(v_00+1)*(theta_00+(1+eps)*np.log(v_00+1))
    dudt=v_00
    dvdt= (gamma_mu**2)*(u_n1-2*u_00+u_p1) \
                    -(gamma_lambda**2)*u_00 -((gamma_mu**2)/(xi))*(f+theta_00+np.log(v_00+1))


    k21=dthetadt[:];
    k22=dudt[:];
    k23=dvdt[:];

    theta_00=theta_00+0.5*k21*dt;
    u_00=u_00+0.5*k22*dt;
    v_00=v_00+0.5*k23*dt;

    #------------
    # K3
    #------------
    u_p1[0:-1]=u_00[1:]
    u_p1[-1]=u_00[0]

    u_n1[0]=u_00[1]
    u_n1[1:]=u_00[0:-1]

    dthetadt=-(v_00+1)*(theta_00+(1+eps)*np.log(v_00+1))
    dudt=v_00
    dvdt= (gamma_mu**2)*(u_n1-2*u_00+u_p1) \
                    -(gamma_lambda**2)*u_00 -((gamma_mu**2)/(xi))*(f+theta_00+np.log(v_00+1))


    k31=dthetadt[:];
    k32=dudt[:];
    k33=dvdt[:];


    theta_00=theta_00+0.5*k31*dt;
    u_00=u_00+0.5*k32*dt;
    v_00=v_00+0.5*k33*dt;

    #------------
    # K4
    #------------
    u_p1[0:-1]=u_00[1:]
    u_p1[-1]=u_00[0]

    u_n1[0]=u_00[1]
    u_n1[1:]=u_00[0:-1]

    dthetadt=-(v_00+1)*(theta_00+(1+eps)*np.log(v_00+1))
    dudt=v_00
    dvdt= (gamma_mu**2)*(u_n1-2*u_00+u_p1) \
                    -(gamma_lambda**2)*u_00 -((gamma_mu**2)/(xi))*(f+theta_00+np.log(v_00+1))


    k41=dthetadt[:];
    k42=dudt[:];
    k43=dvdt[:];


    theta_out=theta_in+(dt/6)*(k11+2*k21+2*k31+k41);
    u_out=u_in+(dt/6)*(k12+2*k22+2*k32+k42);
    v_out=v_in+(dt/6)*(k13+2*k23+2*k33+k43);

    return theta_out,u_out,v_out

def rk4_bk_1d_ensemble(N,n_mem,theta_in,u_in,v_in,dt,q_vector):
    
    # parameters
    eps=0.3
    xi=0.5
    gamma_lambda=np.sqrt(0.2)
    gamma_mu=0.5
    f=3.2
    
    #Initial conditions
    
    theta_00=theta_in[:,:]
    v_00=v_in[:,:]
    u_00=u_in[:,:]

    u_p1=np.zeros((N,n_mem))
    u_p1[0:-1]=u_00[1:,:]
    u_p1[-1]=u_00[0,:]

    u_n1=np.zeros((N,n_mem))
    u_n1[0]=u_00[1,:]
    u_n1[1:]=u_00[0:-1,:]

    #------------
    # K1
    #------------
    dthetadt=-(v_00+1)*(theta_00+(1+eps)*np.log(v_00+1))
    dudt=v_00
    dvdt= (gamma_mu**2)*(u_n1-2*u_00+u_p1) \
                    -(gamma_lambda**2)*u_00 -((gamma_mu**2)/(xi))*(f+theta_00+np.log(v_00+1))+q_vector

    k11=dthetadt[:,:];
    k12=dudt[:,:];
    k13=dvdt[:,:];

    theta_00=theta_00+0.5*k11*dt;
    u_00=u_00+0.5*k12*dt;
    v_00=v_00+0.5*k13*dt;

    #------------
    # K2
    #------------
    u_p1[0:-1,:]=u_00[1:,:]
    u_p1[-1,:]=u_00[0,:]

    u_n1[0,:]=u_00[1,:]
    u_n1[1:,:]=u_00[0:-1,:]

    dthetadt=-(v_00+1)*(theta_00+(1+eps)*np.log(v_00+1))
    dudt=v_00
    dvdt= (gamma_mu**2)*(u_n1-2*u_00+u_p1) \
                    -(gamma_lambda**2)*u_00 -((gamma_mu**2)/(xi))*(f+theta_00+np.log(v_00+1))+q_vector


    k21=dthetadt[:,:];
    k22=dudt[:,:];
    k23=dvdt[:,:];

    theta_00=theta_00+0.5*k21*dt;
    u_00=u_00+0.5*k22*dt;
    v_00=v_00+0.5*k23*dt;

    #------------
    # K3
    #------------
    u_p1[0:-1,:]=u_00[1:,:]
    u_p1[-1,:]=u_00[0,:]

    u_n1[0,:]=u_00[1,:]
    u_n1[1:,:]=u_00[0:-1,:]

    dthetadt=-(v_00+1)*(theta_00+(1+eps)*np.log(v_00+1))
    dudt=v_00
    dvdt= (gamma_mu**2)*(u_n1-2*u_00+u_p1) \
                    -(gamma_lambda**2)*u_00 -((gamma_mu**2)/(xi))*(f+theta_00+np.log(v_00+1))+q_vector


    k31=dthetadt[:,:];
    k32=dudt[:,:];
    k33=dvdt[:,:];


    theta_00=theta_00+0.5*k31*dt;
    u_00=u_00+0.5*k32*dt;
    v_00=v_00+0.5*k33*dt;

    #------------
    # K4
    #------------
    u_p1[0:-1,:]=u_00[1:,:]
    u_p1[-1,:]=u_00[0,:]

    u_n1[0,:]=u_00[1,:]
    u_n1[1:,:]=u_00[0:-1,:]

    dthetadt=-(v_00+1)*(theta_00+(1+eps)*np.log(v_00+1))
    dudt=v_00
    dvdt= (gamma_mu**2)*(u_n1-2*u_00+u_p1) \
                    -(gamma_lambda**2)*u_00 -((gamma_mu**2)/(xi))*(f+theta_00+np.log(v_00+1))+q_vector


    k41=dthetadt[:,:];
    k42=dudt[:,:];
    k43=dvdt[:,:];


    theta_out=theta_in+(dt/6)*(k11+2*k21+2*k31+k41);
    u_out=u_in+(dt/6)*(k12+2*k22+2*k32+k42);
    v_out=v_in+(dt/6)*(k13+2*k23+2*k33+k43);

    return theta_out,u_out,v_out
    
# NOTE: There are problems in this calculation because a variable
# and a function have the same names
def grad_log_post(H,R,R_inv,y,y_i,B,x_s_i,x0_mean):
    #obs_part=(H.transpose()).dot(R_inv).dot(y-y_i)[:,0]
    obs_part=B.dot(H.transpose()).dot(R_inv).dot(y-y_i)[:,0]
    #prior_part=B_inv.dot(x_s_i-x0_mean)
    prior_part=x_s_i-x0_mean
    grad_log_post_est=obs_part-prior_part
    #return grad_log_post_est,obs_part,prior_part;
    return grad_log_post_est;

def pff(n_mem,n_states,ensemble,obs_vect,index_obs):
    
    B=np.cov(ensemble)
    x0_mean=np.mean(ensemble,axis=1)
    
    r_influ=5
    B=localization(r_influ,N,B)
        
    # Pseudo-time flow
    s=0
    max_s=100
    #ds=0.1
    ds=0.05/10;
    alpha=0.05/10 #Tuning parameter for the covariance of the kernel

    x_s=np.zeros((n_states,n_mem),order='F');
    x_s=ensemble.copy()

    kernel=np.zeros((n_states,n_mem,n_mem),order='F')
    dkdx=np.zeros((n_states,n_mem,n_mem),order='F')

    # Pseudoflow 
    python_pseudoflow=np.zeros((n_states,n_mem,max_s+1))
    python_pseudoflow[:,:,0]=x_s.copy()

    n_obs=np.sum(obs_vect > -999)
    # Pseudo time for data assimilation
    while s < max_s:

        H    = np.zeros((n_obs, n_states));        # the ensemble in obs space 
        Hx   = np.zeros((n_obs, n_mem));           # the ensemble in obs space 
        dHdx = np.zeros((n_obs, n_states, n_mem)); # the adjoint of obs operator    

        gradient_posterior=np.zeros((n_states,n_mem))
        dpdx=np.zeros((n_states,n_mem))

        # Observation Operator
        for i in range(n_mem):

            H=h_operator(n_states,obs_vect)
            
            #x_s[:,i]=python_pseudoflow[:,i,s]
            #Hx[:,i]=np.ones((n_obs,1))[:,0] 
            #Hx[0,:]=x_s[10,:]
            #Hx[1,:]=x_s[511,:]
            
            Hx[:,:]=x_s[index_obs,:]   
#             Hx[1,:]=x_s[1,:]
#             Hx[2,:]=x_s[2,:]

            dHdx[:,:,i]=np.ones((n_obs,n_states))

            y=np.ones((n_obs,1))
            #y[:,0]=obs_vect[:,0]
            y[:,0]=obs_vect[index_obs]

            y_i=np.ones((n_obs,1))
#             y_i[0,0]=Hx[0,i]
#             y_i[1,0]=Hx[1,i]
            y_i[:,0]=Hx[:,i]
#             y_i[1,0]=Hx[1,i]
#             y_i[2,0]=Hx[2,i]

    #         if s==0:
    #             x_s[:,i]=x0[:,i]
    #         else
            #grad_log_post[:,i]=(Hx.transpose()).dot(np.linalg.inv(R)).dot(y-y_i)-np.linalg.inv(B).dot(x_s[:,i]-x0_mean)
            dpdx[:,i]=grad_log_post(H,R,R_inv,y,y_i,B,x_s[:,i],x0_mean);
            
            # NOTE: There are problems in this calculation because a variable
            # and a function have the same names

            #print('It passed this line - check 01')

            # Kernel calculation

        B_d=np.zeros((n_states))
        for d in range(n_states):
            B_d[d]=B[d,d]

        #print(f'd={d}')

        fs=np.zeros((n_states,n_mem))
        I_f=np.zeros((n_states,n_mem)) # important to make sure grad_KL starts from zero! This was the error vs Matlab
        attractive_term=np.zeros((n_states,n_mem))
        repelling_term=np.zeros((n_states,n_mem))

        for i in range(n_mem):

            for j in range(n_mem):
                #xj=0;
    #             dpdx[:,j]=gradient_posterior[:,j];
                if j>=i:
                    kernel[:,i,j]=np.exp((-1/2)*((x_s[:,i]-x_s[:,j])**2)/(alpha*B_d[:]));
                    #dkdx[:,i,j]=((x_s[:,i]-x_s[:,j])/(alpha*B_d[:]))*kernel[:,i,j];
                    dkdx[:,i,j]=((x_s[:,i]-x_s[:,j])/(alpha))*kernel[:,i,j];
                else:
                    kernel[:,i,j]=kernel[:,j,i];
                    dkdx[:,i,j]=-dkdx[:,j,i];

                attractive_term[:,i]=(1/n_mem)*(kernel[:,i,j]*dpdx[:,j])
                repelling_term[:,i]=(1/n_mem)*(dkdx[:,i,j])

                I_f[:,i]=I_f[:,i]+attractive_term[:,i]+repelling_term[:,i];
                #I_f[:,i]=I_f[:,i]+attractive_term[:,i];


            # Update the state vector for next pseudo time step

            # Kernel evolution
    #         pff_kernel[:,:,:,s]=kernel[:,:,:]
    #         pff_dkdx[:,:,:,s]=dkdx[:,:,:]
            # Gradient posterior evolution
    #         pff_grad_log_post[:,:,s]=dpdx[:,:]
            # Gradient KL divergence evolution
    #         pff_grad_KL[:,:,s]=I_f
    #         pff_grad_KL_attractive[:,:,s]=attractive_term
    #         pff_grad_KL_repelling[:,:,s]=repelling_term

        for i in range(n_mem):
            #fs[:,i]=B.dot(I_f[:,i])
            fs[:,i]=I_f[:,i]
            x_s[:,i]=x_s[:,i]+ds*fs[:,i]
        python_pseudoflow[:,:,s+1]=x_s

        #print(f'finished with s={s}')

        s=s+1
            #python_pseudoflow[:,:,s]=x_s
        
        posterior_vect=python_pseudoflow[:,:,-1]
        mean_posterior=np.mean(posterior_vect, axis=1)
        cov_posterior=np.cov(posterior_vect)
            
        pff_pseudoflow={"posterior":posterior_vect,"mean_post":mean_posterior,"cov_post":cov_posterior}
    return  pff_pseudoflow


name_exp='pff_loc'

# Test for Burridge-Knopoff RSF 1D Ensemble Forward Model 
# 2023/09/27
# by Hamed Ali Diab-Montero
# h.a.diabmontero@tudelft.nl\

# TRUTH
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
total_time=5000;
nt=int(total_time/tstep)
warm_nt = int(200/tstep)                 # number of warm-up time steps



folder_truth='/home/hamed/Data Assimilation Methods/data/bk_rsf_1d/BK_RSF_1D_datasets_chaotic/truth/'

filename_data_theta_truth=os.path.join(folder_truth,'truth_theta_bk1d_chaotic.txt')
filename_data_v_truth=os.path.join(folder_truth,'truth_v_bk1d_chaotic.txt')
filename_data_u_truth=os.path.join(folder_truth,'truth_u_bk1d_chaotic.txt')
filename_data_tau_truth=os.path.join(folder_truth,'truth_tau_bk1d_chaotic.txt')

filename_time_truth=os.path.join(folder_truth,'time_truth_bk1d_chaotic.txt')

theta_truth=np.genfromtxt(filename_data_theta_truth)
v_truth=np.genfromtxt(filename_data_v_truth)
u_truth=np.genfromtxt(filename_data_u_truth)
tau_truth=np.genfromtxt(filename_data_tau_truth)
t_truth=np.genfromtxt(filename_time_truth)

print('Finished loading the truth')


# Observations

obs_rate = 225     # interval of time steps between observations
obs_den = 1       # observation density (every obs_den -th grids are observed)
t_obs = t_truth[::obs_rate]
index_truth = range(len(t_truth))

t_first_da = int(warm_nt/obs_rate)+1 # index first assimilation 

folder_obsnet='/home/hamed/Data Assimilation Methods/data/bk_rsf_1d/BK_RSF_1D_datasets_chaotic/obsnet/'
filename_time_obsnet_c3=os.path.join(folder_obsnet,'obsnet_c3','obsnet_time_bk1d_chaotic_obs_c3.txt')
filename_tau_obsnet_c3=os.path.join(folder_obsnet,'obsnet_c3','obsnet_tau_bk1d_chaotic_obs_c3.txt')
filename_theta_obsnet_c3=os.path.join(folder_obsnet,'obsnet_c3','obsnet_theta_bk1d_chaotic_obs_c3.txt')
filename_u_obsnet_c3=os.path.join(folder_obsnet,'obsnet_c3','obsnet_u_bk1d_chaotic_obs_c3.txt')
filename_v_obsnet_c3=os.path.join(folder_obsnet,'obsnet_c3','obsnet_vel_bk1d_chaotic_obs_c3.txt')

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
    
q_error=0.5
q_vector=np.random.normal(0,q_error,n_mem*N)
q_vector=np.reshape(q_vector,(N,n_mem))


# ctlmean = X_truth[:, 0] + np.random.multivariate_normal(np.zeros(n_x)+1.5*np.ones(n_x), np.eye(n_x)).T

# initial condition
n_t=len(t_truth)
X_t = np.zeros((n_x*N, n_mem, n_t))
X_t_expanded=np.zeros(((n_x+1)*N,n_mem,n_t))
# Q = 2 * np.eye(n_x)            # background error covariance (only for the initial perturbation)
# Q_inv = np.linalg.inv(Q)
# X_t_2= np.random.multivariate_normal(ctlmean, Q, n_mem).T

folder_prior='/home/hamed/Data Assimilation Methods/data/bk_rsf_1d/BK_RSF_1D_datasets_periodic/prior/'
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
# We will assimilate this number of steps
# t_assim < len(t_truth)-obsrate 
# t_assim = len(t_truth)-50
#t_assim = int((warm_nt*tstep+n_t*tstep-500)/tstep)
#t_assim = int((t_obs[-1]-1)/tstep)
t_assim = len(t_truth)-obs_rate 

for k in range(warm_nt):
    X_t[0:N,:,k+1],X_t[N:2*N,:,k+1],X_t[2*N:3*N,:,k+1] = rk4_bk_1d_ensemble(N,n_mem,X_t[0:N,:,k],X_t[N:2*N,:,k],X_t[2*N:3*N,:,k],tstep,q_vector)
    X_t[3*N:4*N,:,k+1]=f+X_t[0*N:1*N,:,k+1]+np.log(X_t[2*N:3*N,:,k+1]+1)

    X_t_expanded[0:3*N,:,k+1]=X_t[0:3*N,:,k+1].copy()
    X_t_expanded[3*N:4*N,:,k+1]=X_t[3*N:4*N,:,k+1].copy()
    X_t_expanded[4*N:5*N,:,k+1]=q_vector.copy()
    
    
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
        X_t[0:N,:,k],X_t[N:2*N,:,k],X_t[2*N:3*N,:,k] = rk4_bk_1d_ensemble(N,n_mem,X_t[0:N,:,k-1],X_t[N:2*N,:,k-1],X_t[2*N:3*N,:,k-1],tstep,q_vector)
        X_t[3*N:4*N,:,k]=f+X_t[0*N:1*N,:,k]+np.log(X_t[2*N:3*N,:,k]+1)
    
        X_t_expanded[0:3*N,:,k]=X_t[0:3*N,:,k].copy()
        X_t_expanded[3*N:4*N,:,k]=X_t[3*N:4*N,:,k].copy()
        X_t_expanded[4*N:5*N,:,k]=q_vector.copy()


    if(np.min(X_t[2*N:3*N,:,:])<=-1):
        print(f'Warning: A velocity below the threshold at d: {d} from forward modeling')

    t = t_analysis
        
    var_rank=10
    tau_ensemble=f+X_t[var_rank,:,t]+np.log(X_t[2*N+var_rank,:,t]+1)
    list_ensemble_tau= tau_ensemble.tolist() # To do the histogram on the prior
    list_ensemble_v=X_t[2*N+var_rank,:,t].tolist() # To do the histogram on the prior
    
    # Observations vectors
    y_t = np.ones(((n_x+1)*N, 1))*-999
    index_obs = np.where(v_obsnet[d, :] > -999)[0] # Experiment specific
    num_obs=len(index_obs)
    #y_t[:, 0] = x_obset[d, 2:] # Experiment specific
    #y_t[0:N, 0][index_obs] t_analysis = int(np.ceil(t_obs[d]/tstep))= theta_obsnet[d,:][index_obs] # Observations theta
    #y_t[N:2*N, 0][index_obs] = u_obsnet[d,:][index_obs] # Observations slip
    y_t[2*N:3*N, 0][index_obs] = v_obsnet[d,:][index_obs] # Observations slip-rate
    y_t[3*N:4*N, 0][index_obs] = tau_obsnet[d,:][index_obs] # Observations slip-rate

    index_pff= np.where(y_t[:, 0] > -999)[0]
    
    # Prior ensemble for data assimilation
    X_da_prior=X_t_expanded[:, :, t]
    X_da_prior[2*N:3*N,:]=np.log(X_da_prior[2*N:3*N, :]+1)
    
    list_prior.append(X_da_prior)
    list_preupdate.append(X_t[:, :, t].copy())
    # Ensemble Kalman Filter part of the method
    # We need to access t-1 because of the inx system of python
    #post_enkf = enkf(n_mem, n_x*N, X_t[:, :, t-1], y_t, R)
    post_pff = pff(n_mem, (n_x+1)*N, X_da_prior, y_t[:,0], index_pff)

    X_ens=post_pff['posterior']
    X_ens_mean=post_pff['mean_post']
    P_ens=post_pff['cov_post']
    
#     s_matrix=post_enkf['sens_matrix']    
    
#     list_sensitivity.append(s_matrix)
#     list_diag_dfs.append(np.diag(s_matrix))
#     list_iga.append(np.matrix.trace(s_matrix)/num_obs)
#     K_gain_t[:, :, d]=post_enkf['kalman_gain']
    # We need to access t-1 because of the inx system of python
    
    # Posterior ensemble for data assimilation
    X_da_posterior= X_ens.copy()
    
    # Adding prior, obs, posterior
    list_obs.append(y_t)
    list_posterior.append(post_pff['posterior'])
    
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
    # Update q_vector
    q_vector=X_da_posterior[4*N:5*N, :]
    
    
    list_update.append(X_t[:, :, t].copy())
    
    #print(q_vector.shape)
    
    P_t = P_ens   
    
    if(np.min(X_t[2*N:3*N,:,:])<=-1):
        print(f'Warning: A velocity below the threshold at d: {d} from data assimilation')
    
        #Rank Histogram
    if(d<v_obsnet.shape[0]):
        #var_rank=10
        
#         if(np.max(X_ens[var_rank,:])>x_rank_truth[d,var_rank]):
#             more+=1
#         elif(np.min(X_ens[var_rank,:])<x_rank_truth[d,var_rank]):
#             less+=1
#         else:
#             inside+=1
        
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
            
        #list_comparison_truth.append(x_rank_truth[d,int(var_rank/obs_den)])
        #list_comparison_ens.append(X_ens[var_rank,:])
        list_assim.append(t_obs[d])
        
    
    
    d += 1
    
    print(f'Finished with obs: {d}')
    
samples_v=sample_histogram(n_mem, rank_histogram_v, n_samples=1000)
samples_tau=sample_histogram(n_mem, rank_histogram_tau, n_samples=1000)
    
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

filename_estimate_theta='./results_rmse/rmse_theta_bk1d_model_error_obs_den_'+str(obs_den).zfill(2)+'_run_'+name_exp+'.txt'
np.savetxt(filename_estimate_theta,rmse_theta)

#--------------------------------
# SLIP
#--------------------------------
index_variable=1
error_u=u_truth[:,:]-np.transpose(np.mean(X_t[index_variable*N:(index_variable+1)*N,:,:],axis=1))
error_u_squared=error_u**2
rmse_u=np.sqrt((1/N)*np.sum(error_u_squared,axis=1))

filename_estimate_u='./results_rmse/rmse_u_bk1d_model_error_obs_den_'+str(obs_den).zfill(2)+'_run_'+name_exp+'.txt'
np.savetxt(filename_estimate_u,rmse_u)


#--------------------------------
# VELOCITY
#--------------------------------
index_variable=2
error_v=v_truth[:,:]-np.transpose(np.mean(X_t[index_variable*N:(index_variable+1)*N,:,:],axis=1))
error_v_squared=error_v**2
rmse_v=np.sqrt((1/N)*np.sum(error_v_squared,axis=1))

filename_estimate_v='./results_rmse/rmse_v_bk1d_model_error_obs_den_'+str(obs_den).zfill(2)+'_run_'+name_exp+'.txt'
np.savetxt(filename_estimate_v,rmse_v)


#--------------------------------
# TAU
#--------------------------------
index_variable=3
error_tau=tau_truth[:,:]-np.transpose(np.mean(X_t[index_variable*N:(index_variable+1)*N,:,:],axis=1))
error_tau_squared=error_tau**2
rmse_tau=np.sqrt((1/N)*np.sum(error_tau_squared,axis=1))

filename_estimate_tau='./results_rmse/rmse_tau_bk1d_model_error_obs_den_'+str(obs_den).zfill(2)+'_run_'+name_exp+'.txt'
np.savetxt(filename_estimate_tau,rmse_tau)

# ----------------------------
#  RANK HISTOGRAM
# ----------------------------

rank_hist_summary=np.zeros((n_mem+1,2))
rank_hist_summary[:,0]=range(n_mem+1)
rank_hist_summary[:,1]=rank_histogram_v
rank_hist_summary[:,1]=rank_histogram_tau

filename_rank_hist='./results_rank_hist/rank_hist_bk1d_model_error_obs_den_'+str(obs_den).zfill(2)+'_run_'+name_exp+'.txt'
np.savetxt(filename_rank_hist,rank_hist_summary)

filename_samples_v='./results_rank_hist/samples_v_hist_bk1d_model_error_obs_den_'+str(obs_den).zfill(2)+'_run_'+name_exp+'.txt'
np.savetxt(filename_samples_v,samples_v)

filename_samples_tau='./results_rank_hist/samples_tau_hist_bk1d_model_error_obs_den_'+str(obs_den).zfill(2)+'_run_'+name_exp+'.txt'
np.savetxt(filename_samples_tau,samples_tau)

print('Finished with the RANK HISTOGRAM saving files')



