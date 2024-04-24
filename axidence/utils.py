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
