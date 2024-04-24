import numpy as np
import strax
import straxen


def grouped_peak_dtype(n_channels=straxen.n_tpc_pmts):
    dtype = strax.peak_dtype(n_channels=n_channels)
    # since event_number is int64 in event_basics
    dtype += [
        (("Group number of peaks", "group_number"), np.int64),
    ]
    return dtype
