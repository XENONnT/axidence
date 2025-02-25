from strax.plugin import Plugin


class ExhaustPlugin(Plugin):
    """Plugin that exhausts all chunks when fetching data."""
    chunk_number = None

    def _fetch_chunk(self, d, iters, check_end_not_before=None):
        while super()._fetch_chunk(d, iters, check_end_not_before=check_end_not_before):
            pass
        return False

    def do_compute(self, chunk_i=None, **kwargs):
        if self.chunk_number is None:
            first_chunk = 0
        else:
            first_chunk = self.chunk_number[0]

        if chunk_i != first_chunk:
            raise RuntimeError(
                f"{self.__class__.__name__} is an ExhaustPlugin. "
                "It should read all chunks together can process them together."
            )
        return super().do_compute(chunk_i=chunk_i, **kwargs)