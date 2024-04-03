import numpy as np

def L96_RK4(X, dt, F):
    """
    Perform a single RK4 step for the Lorenz 96 model.
    
    Args:
    X (numpy.array): Current state vector.
    dt (float): Time step for integration.
    F (float): Forcing term in the Lorenz 96 model.

    Returns:
    numpy.array: Updated state vector after one RK4 step.
    """
    # Initialize temporary variables for RK4 calculations
    dim = len(X)
    k1 = np.zeros(dim)
    k2 = np.zeros(dim)
    k3 = np.zeros(dim)
    k4 = np.zeros(dim)
    X_out = np.zeros(dim)

   # Calculate k1
    tmp_b = X.copy()  # before integration
    
    X_p1 = np.concatenate((tmp_b[1:], [tmp_b[0]]))
    X_00 = tmp_b
    X_n1 = np.concatenate(([tmp_b[-1]], tmp_b[:-1]))
    X_n2 = np.concatenate((tmp_b[-2:], tmp_b[:-2]))
    
    k1 = (X_p1 - X_n2) * X_n1 - X_00 + F

    # Subsequent RK4 calculations (k2, k3, k4) follow similar steps
   # Calculate k2    
    tmp_b = X + 0.5 * k1 * dt
    X_p1 = np.concatenate((tmp_b[1:], [tmp_b[0]]))
    X_00 = tmp_b
    X_n1 = np.concatenate(([tmp_b[-1]], tmp_b[:-1]))
    X_n2 = np.concatenate((tmp_b[-2:], tmp_b[:-2]))
    
    k2 = (X_p1 - X_n2) * X_n1 - X_00 + F

   # Calculate k3    
    tmp_b = X + 0.5 * k2 * dt
    X_p1 = np.concatenate((tmp_b[1:], [tmp_b[0]]))
    X_00 = tmp_b
    X_n1 = np.concatenate(([tmp_b[-1]], tmp_b[:-1]))
    X_n2 = np.concatenate((tmp_b[-2:], tmp_b[:-2]))
    
    k3 = (X_p1 - X_n2) * X_n1 - X_00 + F

    # Calculate k4
    tmp_b = X + k3 * dt
    X_p1 = np.concatenate((tmp_b[1:], [tmp_b[0]]))
    X_00 = tmp_b
    X_n1 = np.concatenate(([tmp_b[-1]], tmp_b[:-1]))
    X_n2 = np.concatenate((tmp_b[-2:], tmp_b[:-2]))
    
    k4 = (X_p1 - X_n2) * X_n1 - X_00 + F
    
    X_out = X + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return X_out

def calc_X(X):
    """
    Calculate shifted state vectors needed for RK4 steps in the Lorenz '96 model.

    Args:
    X (numpy.array): Current state vector.

    Returns:
    tuple: Shifted state vectors (X_p1, X_00, X_n1, X_n2).
    """    
    
    # Calculate X_p1, X_00, X_n1, and X_n2
    X_p1 = np.roll(X, -1, axis=0)
    X_00 = X.copy()
    X_n1 = np.roll(X, 1, axis=0)
    X_n2 = np.roll(X, 2, axis=0)
    return X_p1, X_00, X_n1, X_n2

def L96_RK4_ensemble(X_in, dt, F):
    """
    Integrate an ensemble of Lorenz '96 model states using the RK4 method.

    Args:
    X_in (numpy.ndarray): Input array of ensemble states (dim, n_mem).
    dt (float): Time step for integration.
    F (float): Forcing term in the Lorenz '96 model.

    Returns:
    numpy.ndarray: Updated ensemble states after one RK4 step.
    """
    
    dim, n_mem = X_in.shape
    k1 = np.zeros((dim, n_mem))
    k2 = np.zeros((dim, n_mem))
    k3 = np.zeros((dim, n_mem))
    k4 = np.zeros((dim, n_mem))
    X_out = np.zeros((dim, n_mem))

    tmp_b = X_in.copy()  # before integration
    
    X_p1, X_00, X_n1, X_n2 = calc_X(tmp_b)

    # Calculate k1
    k1 = (X_p1 - X_n2) * X_n1 - X_00 + F

    tmp_b2 = X_in + 0.5 * k1 * dt
    X_p1, X_00, X_n1, X_n2 = calc_X(tmp_b2)

    # Calculate k2
    k2 = (X_p1 - X_n2) * X_n1 - X_00 + F

    tmp_b3 = X_in + 0.5 * k2 * dt
    X_p1, X_00, X_n1, X_n2 = calc_X(tmp_b3)

    # Calculate k3
    k3 = (X_p1 - X_n2) * X_n1 - X_00 + F

    tmp_b4 = X_in + k3 * dt
    X_p1, X_00, X_n1, X_n2 = calc_X(tmp_b4)

    # Calculate k4
    k4 = (X_p1 - X_n2) * X_n1 - X_00 + F

    X_out = X_in + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return X_out
