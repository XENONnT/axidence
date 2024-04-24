def copy_dtype(dtype_reference, required_names):
    """
    Copy dtype from dtype_reference according to required_names.
    Args:
        dtype_reference (list): dtype reference to copy from
        required_names (list or tuple): names to copy

    Returns:
        list: copied dtype
    """
    dtype = []
    for n in required_names:
        for x in dtype_reference:
            found = False
            if (x[0][1] == n) and (not found):
                dtype.append(x)
                found = True
                break
        if not found:
            raise ValueError(f"Could not find {n} in dtype_reference!")
    return dtype


def needed_dtype(deps, dependencies_by_kind, func):
    # intersection depends_on's dtype.names will be needed in event building
    needed_fields = func(
        *tuple(
            set.union(*tuple(set(deps[d].dtype_for(d).names) for d in dk))
            for dk in dependencies_by_kind().values()
        )
    )
    dtype_reference = list(
        func(
            *tuple(
                set.union(*tuple(set(deps[d].dtype_for(d).descr) for d in dk))
                for dk in dependencies_by_kind().values()
            )
        )
    )
    _peaks_dtype = copy_dtype(dtype_reference, needed_fields)
    if len(_peaks_dtype) != len(needed_fields):
        raise ValueError(
            f"Weird! Could not find all needed fields {needed_fields} in {dtype_reference}!"
        )
    return needed_fields, _peaks_dtype
