#A mathematical model for the atomic clock error
#L Galleani et al 2003 Metrologia 40 S257
#doi 10.1088/0026-1394/40/3/305

import numpy as np
from typing import Union


#TODO Potentially OOP, a clock class


def stochastic_clock(tp : float, 
                     N : float, 
                     X0 : np.ndarray = np.array([0, 0]), 
                     mu : np.ndarray = np.array([1, 0]), 
                     sigma : np.ndarray = np.array([0, 0]),
                     method : str = 'Galleani_clock') -> np.ndarray:
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
        the output gives the stochastic timegrid and not only the deviations. 
    sigma : np.ndarray, default np.array([0, 0])
        Diffusion coefficients of the noise components which give the intensity of each
        noise. Deafult zero for a perfect clock.
    method : str, default Galleani_clock
        Choose the method by which to calculate the stochastic clock deviations. 
        Currently only a single one is implemented, but this argument is here to allow
        future functionality expansion.

    Returns
    -------
    timegrid: dict
        Return a dictionary containing the original and stochastic timegrids, and the 
        deviation from the original timegrid.
    '''
    if method == 'Galleani_clock':
        t_stochastic, t_original = Galleani_clock(tp=tp, N=N, X0=X0, mu=mu, sigma=sigma)
        deviation = t_original - t_stochastic 
        timegrid = {'stochastic': t_stochastic, 'original': t_original, 
                    'deviation': deviation}
    return timegrid


def Galleani_clock(tp : float, 
                    N : float, 
                    X0 : np.ndarray, 
                    mu : np.ndarray, 
                    sigma : np.ndarray) -> tuple:
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
    BM = np.dot(B, mu)

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


def clock_error(timegrid : np.ndarray, 
                timegrid_stochastic : np.ndarray, 
                amplitudes : np.ndarray) -> tuple:
    '''
    This is not a statistical measure, rather for viewing the deviations between the
    two timegrids. The timegrid across which the signals are compared is the truncated
    union of the stochastic and non-stochastic timesteps.
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


"""
def clock_deviation(timegrid : np.ndarray, 
                    timegrid_stochastic : np.ndarray, 
                    amplitudes : np.ndarray) -> tuple:
    '''
    This is not a statistical measure, rather for viewing the deviations between the
    two timegrids. The timegrid across which the signals are compared is the truncated
    union of the stochastic and non-stochastic timesteps.
    '''
    #Create a new union timegrid that does not exceed the shortest time
    stop_time = np.min([np.max(timegrid), np.max(timegrid_stochastic)])
    time_union = np.union1d(timegrid, timegrid_stochastic)
    time_union = time_union[time_union <= stop_time]

    #Block discretization of signal
    #Necessary to avoid interpolation of medial amplitudes
    t = np.repeat(timegrid, 2)
    t = t[1:]
    t_stochastic = np.repeat(timegrid_stochastic, 2)
    t_stochastic = t_stochastic[1:]
    assert len(t) == len(t_stochastic)
    amp = np.repeat(amplitudes, 2)
    amp = amp[:-1]
    assert len(t) == len(amp)

    #Linear interpolation onto union timegrid
    amplitude_non_stoch = np.interp(time_union, t, amp)
    amplitude_stoch = np.interp(time_union, t_stochastic, amp)

    signal = np.vstack((time_union, amplitude_non_stoch)) 
    signal_stoch = np.vstack((time_union, amplitude_stoch))
    
    return signal, signal_stoch
"""