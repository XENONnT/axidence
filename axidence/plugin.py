import strax
from strax import Plugin


export, __all__ = strax.exporter()


@export
class ExhaustPlugin(Plugin):
    """Plugin that exhausts all chunks when fetching data."""

    def _fetch_chunk(self, d, iters, check_end_not_before=None):
        while super()._fetch_chunk(d, iters, check_end_not_before=check_end_not_before):
            pass
        return False

    def do_compute(self, chunk_i=None, **kwargs):
        if chunk_i != 0:
            raise RuntimeError(
                f"{self.__class__.__name__} is an ExhaustPlugin. "
                "It should read all chunks together can process them together."
            )
        return super().do_compute(chunk_i=chunk_i, **kwargs)


@export
class InferDtypePlugin(Plugin):
    """Plugin that infers the dtype from the dependency dtype."""

    def infer_dtype(self):
        raise NotImplementedError

    def refer_dtype(self):
        raise NotImplementedError

    def copy_dtype(self, dtype_reference, required_names):
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
