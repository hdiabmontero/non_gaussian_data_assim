import numpy as np

def rk4_bk_1d_step(N, theta_in, u_in, v_in, dt):
    """
    Perform a single RK4 step for a one-dimensional system.

    Args:
    N (int): Number of grid points.
    theta_in, u_in, v_in (numpy.array): Input state vectors for theta, u, and v.
    dt (float): Time step for the RK4 update.

    Returns:
    tuple: Updated state vectors (theta_out, u_out, v_out) after one RK4 step.
    """
    # Defining parameters for the equations
    eps = 0.3
    xi = 0.5
    gamma_lambda = np.sqrt(0.2)
    gamma_mu = 0.5
    f = 3.2

    # Setting up initial conditions
    theta_00, u_00, v_00 = theta_in[:], u_in[:], v_in[:]

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

    return theta_out, u_out, v_out

def rk4_bk_1d_ensemble(N, n_mem, theta_in, u_in, v_in, dt):
    """
    Perform a single RK4 step for an ensemble of one-dimensional systems.

    Args:
    N (int): Number of grid points.
    n_mem (int): Number of ensemble members.
    theta_in, u_in, v_in (numpy.array): Input state matrices for theta, u, and v, each with shape (N, n_mem).
    dt (float): Time step for the RK4 update.

    Returns:
    tuple: Updated state matrices (theta_out, u_out, v_out) after one RK4 step.
    """
    # Defining parameters for the equations
    eps = 0.3
    xi = 0.5
    gamma_lambda = np.sqrt(0.2)
    gamma_mu = 0.5
    f = 3.2

    #Initial conditions
    
    # Setting up initial conditions for the ensemble
    theta_00, u_00, v_00 = theta_in[:,:], u_in[:,:], v_in[:,:]
    
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
                    -(gamma_lambda**2)*u_00 -((gamma_mu**2)/(xi))*(f+theta_00+np.log(v_00+1))

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
                    -(gamma_lambda**2)*u_00 -((gamma_mu**2)/(xi))*(f+theta_00+np.log(v_00+1))


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
                    -(gamma_lambda**2)*u_00 -((gamma_mu**2)/(xi))*(f+theta_00+np.log(v_00+1))


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
                    -(gamma_lambda**2)*u_00 -((gamma_mu**2)/(xi))*(f+theta_00+np.log(v_00+1))


    k41=dthetadt[:,:];
    k42=dudt[:,:];
    k43=dvdt[:,:];


    theta_out=theta_in+(dt/6)*(k11+2*k21+2*k31+k41);
    u_out=u_in+(dt/6)*(k12+2*k22+2*k32+k42);
    v_out=v_in+(dt/6)*(k13+2*k23+2*k33+k43);

    return theta_out,u_out,v_out