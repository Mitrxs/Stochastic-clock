#A mathematical model for the atomic clock error
#L Galleani et al 2003 Metrologia 40 S257
#doi 10.1088/0026-1394/40/3/305

import numpy as np


def stochastic_clock(tp, 
                     N , 
                     X0=np.array([0, 0]), 
                     mu=np.array([1, 0]), 
                     sigma=np.array([0, 0]),
                     method='Galleani_exact'):
    '''
    Calculation of the stochastic deviation of timegrids. This function is recommended to 
    being called over `Galleani_clock()`.

    Parameters
    ----------
    tp : float
        Signal length in s.
    N : float 
        Number of signal segments.
    X0 : np.ndarray, default np.array([0, 0])
        Initial conditions of the stochastic process for phase deviation (initial time)
        and random walk component. Initial conditions 0 as we take the initial time and 
        deviation to be zero.
    mu : np.ndarray, default np.array([1, 0])
        Deterministic drift terms for the Wiener processes. Default array is set so that
        the output gives the stochastic timegrid and not only the deviations. For 
        `method="distribution"` this array must be a zero-array.
    sigma : np.ndarray, default np.array([0, 0])
        Diffusion coefficients of the noise components which give the intensity of each
        noise. Deafult zero for a perfect clock.
    method : str, default Galleani_clock
        Choose the method by which to calculate the stochastic clock deviations:

            - Galleani_exact: Use the exact solution to iteratively calculate the 
            deviations of the stochastic timegrid
            - distribution: Calculate the distribution of deviations from their Gaussian 
            at each timepoint (not to be used for simulations).

    Returns
    -------
    timegrid: dict
        Return a dictionary containing the original and stochastic timegrids, and the 
        deviation from the original timegrid.

    #TODO Potentially OOP, a clock class
    '''
    if method == 'Galleani_clock':
        t_stochastic, t_original = Galleani_exact(tp=tp, N=N, X0=X0, 
                                                  mu=mu, sigma=sigma)
        deviation = t_original - t_stochastic 
        timegrid = {'stochastic': t_stochastic, 'original': t_original, 
                    'deviation': deviation}
        
    elif method == 'distribution':
        deviation, t_original = deviation_distribution(tp=tp, N=N, X0=X0, 
                                                         mu=mu, sigma=sigma)
        t_stochastic = deviation + t_original
        timegrid = {'stochastic': t_stochastic, 'original': t_original, 
                    'deviation': deviation}
    return timegrid


def Galleani_exact(tp, N, X0, mu, sigma):
    '''
    Calculation of the stochastic deviation of timegrids.

    Parameters
    ----------
    tp : float
        Signal length in s.
    N : float 
        Number of signal segments.
    X0 : np.ndarray
        Initial conditions of the stochastic process for phase deviation (initial time)
        and random walk component.
    mu : np.ndarray
        Deterministic drift terms for the Wiener processes.
    sigma : np.ndarray
        Diffusion coefficients of the noise components which give the intensity of each
        noise.

    Returns
    -------
    t_stochastic: np.ndarray
        The stochastic timegrid of the signal with shape (N+1,).
    t_original: np.ndarray
        The original equidistant timegrid of the signal with shape (N+1,).
    '''
    
    dt = tp / N #Timestep
    timegrid = np.arange(0, tp+dt, dt) #N+1 elements

    #Calculation matrices
    #eqn. 11, pg 261
    Phi = np.array([[1, dt], [0, 1]])
    B = np.array([[dt, (dt**2) / 2], [0, dt]])
    #BM = np.dot(B, mu)
    m1, m2 = mu
    BM = np.array([(dt * m1) + ( 0.5 * (dt**2) * m2),
                   dt * m2])

    #Mean, covariance for multivariate normal distribution (eqn 17)
    mean = np.zeros(2)
    s1, s2 = sigma
    cov = np.array([[(s1**2)*dt + ((s2**2) * (dt**3))/3, (s2**2) * (dt**2)*0.5], 
                        [(s2**2) * (dt**2)*0.5, (s2**2) * dt]])

    #eqn. 16
    X = np.zeros((2, len(timegrid)))
    X[:,0] = X0.reshape([2])
    for i in range(0, N):
        J = np.random.multivariate_normal(mean, cov) #eqn. 17
        X[:,i+1] = np.dot(Phi, X[:,i]) + BM + J 

    t_stochastic = X[0]

    t_original = timegrid 
    return t_stochastic, t_original


def deviation_distribution(tp, N, X0, mu, sigma):
    '''
    Calculation of the stochastic distribution of deviations from the timegrid at each 
    timepoint.

    Parameters
    ----------
    tp : float
        Signal length in s.
    N : float 
        Number of signal segments.
    X0 : np.ndarray
        Initial conditions of the stochastic process for phase deviation (initial time)
        and random walk component.
    mu : np.ndarray
        Deterministic drift terms for the Wiener processes.
    sigma : np.ndarray
        Diffusion coefficients of the noise components which give the intensity of each
        noise.
    
    Returns
    -------
    deviation: np.ndarray
        The stochastic deviation at each timepoint. An array of shape (N+1,).
    t_original: np.ndarray
        The original equidistant timegrid of the signal with shape (N+1,).
    '''
    #Assign variables
    c1, c2 = X0
    m1, m2 = mu
    s1, s2 = sigma
    dt = tp / N #Timestep
    timegrid = np.arange(0, tp+dt, dt) #N+1 elements
    X = np.zeros((2,len(timegrid)))

    mean = lambda t : np.array([c1 + ((c2 + m1) *t) + (m2 * (t**2) * 0.5),
                                c2 + (m2 * t)])
    cov = lambda t : np.array([[((s1**2) * t) + ((s2**2) * (t**3)/3), (s2**2) * (t**2)/2], 
                               [(s2**2) * (t**2)/2, (s2**2) * t]])
    for i, t in enumerate(timegrid):
        X[:,i] = np.random.multivariate_normal(mean(t), cov(t))
    
    deviation = X[0]
    t_original = timegrid
    #t_stochastic = timegrid + deviation
    return deviation, t_original



def clock_error(timegrid, 
                timegrid_stochastic, 
                amplitudes):
    '''
    This is not a statistical measure, rather for viewing the deviations between the
    two timegrids. The timegrid across which the signals are compared is the truncated
    union of the stochastic and non-stochastic timepoints.

    Parameters
    ----------
    timegrid : np.ndarray
        The original equidistant timegrid of the signal with shape (N+1,).
    timegrid_stochastic: np.ndarray
        The stochastic timegrid of the signal with shape (N+1,).
    amplitudes: np.ndarray
        The amplitudes of the signal.
        
    Returns
    -------
    signal : np.ndarray
        Two-dimensional array of the union timegrid between the original and stochastic
        timegrids (first row), and the indexed signal amplitudes for the original 
        timegrid (second row).
    signal_stochastic: np.ndarray
        Two-dimensional array of the union timegrid between the original and stochastic
        timegrids (first row), and the indexed signal amplitudes for the stochastic 
        timegrid (second row).
    '''

    #Create a new union timegrid that does not exceed the shortest time
    stop_time = np.min([np.max(timegrid), np.max(timegrid_stochastic)])
    time_union = np.union1d(timegrid, timegrid_stochastic)
    time_union = time_union[time_union <= stop_time]

    #Timepoints on deterministic timegrid in time_union = True
    time_deterministic_bool = np.in1d(time_union, timegrid)

    #Index time_union points on the relevant timegrid
    time_index = np.zeros(time_deterministic_bool.shape, dtype=np.int64)
    for i, elem in enumerate(time_union):
        if time_deterministic_bool[i]:
            time_index[i] = np.argwhere(timegrid == elem)[0][0]
        else:
            time_index[i] = np.argwhere(timegrid_stochastic == elem)[0][0]

    #Interpolate the amplitudes of the deterministic and stochastic signals onto the 
    #union timegrid
    amp_deterministic = np.zeros(time_union.shape)
    amp_stochastic = np.zeros(time_union.shape)
    for i, (index, deterministic) in enumerate(zip(time_index, time_deterministic_bool)):
        if deterministic and i != 0:
            #If the timepoint is on the deterministic timegrid, return its corresponding
            #amplitude
            #If the timepoint is on the stochastic timegrid, return the previous amplitude
            amp_deterministic[i] = amplitudes[index]
            amp_stochastic[i] = amp_stochastic[i-1]
        elif not deterministic and i != 0:
            #If the timepoint is on the stochastic timegrid, return its corresponding
            #amplitude
            #If the timepoint is on the stochastic timegrid, return the previous amplitude
            amp_deterministic[i] = amp_deterministic[i-1]
            amp_stochastic[i] = amplitudes[index]

        elif deterministic and i == 0:
            amp_deterministic[i] = amplitudes[index]
            amp_stochastic[i] = amp_deterministic[i]
        elif not deterministic and i == 0:
            amp_stochastic[i] = amplitudes[index]
            amp_deterministic[i] = amp_stochastic[i]

    signal = np.vstack((time_union, amp_deterministic))
    signal_stochastic = np.vstack((time_union, amp_stochastic))

    return signal, signal_stochastic
