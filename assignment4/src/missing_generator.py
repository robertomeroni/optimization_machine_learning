import numpy as np
import math

def produce_missings(time_series, perc_miss, length_miss, seed=None):
    """ This function produces a desired missing value pattern to a
    time series:
    
    INPUT:
        - time_series: pandas time series for a specific sensor
        - perc_miss: percentage of desired missings (e.g. 0.05, 0.1, 0.15, ...)
        - length_miss: burst of missings (e.g. 1, 5, 10, 20, ...)
        
    OUTPUT:
        - indexes_missing: positions in which the missings are produced
        - incomplete_time_series: new time series with missing values (put as NaN)
    """

    if seed is not None:
        np.random.seed(seed)
        
    total_size = len(time_series) # Length of the time series
    N_miss = int(total_size*perc_miss) # Total number of samples lost
    if length_miss > N_miss:
        print("Configuration infeasible")
        return None, None

    indexes = np.arange(total_size) # time series indexes
    indexes = np.arange(total_size + 1 - length_miss) # in this cas at most start index can be len -
    N_start = math.ceil(N_miss/length_miss) # Points where bursts start
    indexes_start = []
    indexes_to_remove = []
    # select starting points one by one
    for i_start in range(N_start):
        # select start of brust and remove indexs for future selections
        start = np.random.choice(indexes, size=1) # random selection of N_start initial points
        indexes_start.append(start)
        # we need to remove possible starting indexes from possible set
        indexes_to_remove.append(start[0])
        for i_burst in range(1, length_miss):
            if (start[0] + i_burst) < total_size:
                indexes_to_remove.append(start[0] + i_burst)

            if (start[0] - i_burst) >= 0:
                indexes_to_remove.append(start[0] - i_burst)

        aux_to_remove = np.array(indexes_to_remove)
        indexes = np.delete(np.arange(total_size), np.ravel(aux_to_remove))

    indexes_missing = []
    # iterate over initial points and save indexes for the bursts
    for ind in indexes_start:
        indexes_missing.append(ind)
        for t_s in range(length_miss-1):
            ind_t = ind + t_s + 1
            # special case
            if (ind_t < total_size) and (len(indexes_missing) < N_miss):
                indexes_missing.append(ind_t)
    indexes_missing = np.array(indexes_missing)

    # introduce nans in specified indexes
    indexes_missing = np.ravel(indexes_missing)
    incomplete_time_series = time_series.copy()
    for ind in indexes_missing:
        incomplete_time_series.at[time_series.index[ind]] = float('nan')

    return indexes_missing, incomplete_time_series
