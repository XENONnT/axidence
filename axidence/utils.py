import numpy as np


def copy_dtype(dtype_reference, required_names):
    """
    Copy dtype from dtype_reference according to required_names.
    Args:
        dtype_reference (list): dtype reference to copy from
        required_names (list or tuple): names to copy

    Returns:
        list: copied dtype
    """
    if not isinstance(required_names, (set, list, tuple)):
        raise ValueError(
            "required_names must be set, list or tuple, "
            f"not {type(required_names)}, got {required_names}!"
        )
    if not isinstance(dtype_reference, list):
        raise ValueError(
            f"dtype_reference must be list, not {type(dtype_reference)}, got {dtype_reference}!"
        )
    dtype = []
    for n in required_names:
        for x in dtype_reference:
            found = False
            if (x[0][1] == n) and (not found):
                dtype.append(x)
                found = True
                break
        if not found:
            raise ValueError(f"Could not find {n} in {dtype_reference}!")
    return dtype


def needed_dtype(deps, dependencies_by_kind, func):
    # intersection depends_on's dtype.names will be needed in event building
    needed_fields = sorted(
        func(
            *tuple(
                set.union(*tuple(set(deps[d].dtype_for(d).names) for d in dk))
                for dk in dependencies_by_kind
            )
        )
    )
    dtype_reference = sorted(
        list(
            func(
                *tuple(
                    set.union(*tuple(set(deps[d].dtype_for(d).descr) for d in dk))
                    for dk in dependencies_by_kind
                )
            )
        )
    )
    _peaks_dtype = copy_dtype(dtype_reference, needed_fields)
    if len(_peaks_dtype) != len(needed_fields):
        raise ValueError(
            f"Weird! Could not find all needed fields {needed_fields} in {dtype_reference}!"
        )
    return needed_fields, np.dtype(_peaks_dtype)


def _pick_fields(field, peaks, peaks_dtype):
    if field in peaks.dtype.names:
        _field = peaks[field]
    else:
        if np.issubdtype(peaks_dtype[field], np.integer):
            _field = np.full(len(peaks), -1)
        else:
            _field = np.full(len(peaks), np.nan)
    return _field


def merge_salted_real(peaks_salted, real_peaks, peaks_dtype):
    # combine peaks_salted and peaks
    _peaks = np.empty(len(peaks_salted) + len(real_peaks), dtype=peaks_dtype)
    for n in _peaks.dtype.names:
        _peaks[n] = np.hstack(
            [_pick_fields(n, peaks_salted, peaks_dtype), _pick_fields(n, real_peaks, peaks_dtype)]
        )
    _peaks = np.sort(_peaks, order="time")
    return _peaks
