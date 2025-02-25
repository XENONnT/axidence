from typing import Generator
import warnings
import sys

import strax
from strax.plugin import Plugin



##
# Plugin which allows to use yield in plugins compute method.
# Allows to chunk down output before storing to disk.
# Only works if multiprocessing is omitted.
##


class DownChunkingPlugin(Plugin):
    """Plugin that merges data from its dependencies."""

    parallel = False

    def __init__(self):
        super().__init__()

        if self.parallel:
            raise NotImplementedError(
                f'Plugin "{self.__class__.__name__}" is a DownChunkingPlugin which '
                "currently does not support parallel processing."
            )

    def _iter_compute(self, chunk_i, **inputs_merged):
        return self.do_compute(chunk_i=chunk_i, **inputs_merged)

    def _fix_output(self, result, start, end, superrun, subruns, _dtype=None):
        """Wrapper around _fix_output to support the return of iterators."""
        if not isinstance(result, Generator):
            raise ValueError(
                f"Plugin {self.__class__.__name__} should return a generator in compute method."
            )

        for _result in result:
            if isinstance(_result, dict):
                values = _result.values()
            else:
                if self.multi_output:
                    raise ValueError(
                        f"{self.__class__.__name__} is multi-output and should "
                        "provide a generator of dict output."
                    )
                values = [_result]
            if not all(isinstance(v, strax.Chunk) for v in values):
                raise ValueError(
                    f"Plugin {self.__class__.__name__} should yield (dict of) "
                    "strax.Chunk in compute method."
                )
            yield self.superrun_transformation(_result, superrun, subruns)
    
    def do_compute(self, chunk_i=None, **kwargs):
        """Wrapper for the user-defined compute method.

        This is the 'job' that gets executed in different processes/threads during multiprocessing

        """
        for k, v in kwargs.items():
            if not isinstance(v, strax.Chunk):
                raise RuntimeError(
                    f"do_compute of {self.__class__.__name__} got a {type(v)} "
                    f"instead of a strax Chunk for {k}"
                )

        if len(kwargs):
            # Check inputs describe the same time range
            tranges = {k: (v.start, v.end) for k, v in kwargs.items()}
            start, end = list(tranges.values())[0]

            # For non-saving plugins, don't be strict, just take whatever
            # endtimes are available and don't check time-consistency
            # Side mark this wont work for a plugin which has a SaveWhen.NEVER and other
            # SaveWhen type.
            if hasattr(self.save_when, "values"):
                save_when = max([int(save_when) for save_when in self.save_when.values()])
            else:
                save_when = self.save_when
            if save_when <= strax.SaveWhen.EXPLICIT:
                # </start>This warning/check will be deleted, see UserWarning
                if len(set(tranges.values())) != 1:
                    start = min([v.start for v in kwargs.values()])
                    end = max([v.end for v in kwargs.values()])
                    message = (
                        "New feature, we are ignoring inconsistent the "
                        "possible ValueError in time ranges for "
                        f"{self.__class__.__name__} of inputs: {tranges} "
                        "because this occurred in a save_when.NEVER "
                        "plugin. Report any findings in "
                        "https://github.com/AxFoundation/strax/issues/247"
                    )
                    warnings.warn(message, UserWarning)
                # This block will be deleted </end>
            elif len(set(tranges.values())) != 1:
                message = (
                    f"{self.__class__.__name__} got inconsistent time ranges of inputs: {tranges}"
                )
                raise ValueError(message)
        else:
            # This plugin starts from scratch
            start, end = None, None

        # Save superrun and subruns of chunks in kwargs for further usage
        superrun = self._check_subruns_uniqueness(
            kwargs, {k: v.superrun for k, v in kwargs.items()}
        )
        subruns = self._check_subruns_uniqueness(kwargs, {k: v.subruns for k, v in kwargs.items()})

        _kwargs = {k: v.data for k, v in kwargs.items()}
        if self.compute_takes_chunk_i:
            _kwargs["chunk_i"] = chunk_i
        if self.compute_takes_start_end:
            _kwargs["start"] = start
            _kwargs["end"] = end
        result = self.compute(**_kwargs)
        del _kwargs

        if self.clean_chunk_after_compute:
            # Free memory by deleting the input chunks
            keys = list(kwargs.keys())
            for k in keys:
                # Minus one accounts for reference created by sys.getrefcount itself
                n = sys.getrefcount(kwargs[k].data) - 1
                if n != 1:
                    raise ValueError(
                        f"Reference count of input {k} is {n} "
                        "and should be 1. This is a memory leak."
                    )
                del kwargs[k].data
        return self._fix_output(result, start, end, superrun, subruns)
    
    @staticmethod
    def _check_subruns_uniqueness(kwargs, subrunses):
        """Check if the subruns of the all inputs are the same."""
        _subrunses = list(subrunses.values())
        if not all(_subruns == _subrunses[0] for _subruns in _subrunses):
            raise ValueError(
                "Computing inputs' superruns or subrunses of "
                f"{kwargs} are different: {subrunses}."
            )
        if len(subrunses) == 0:
            # The plugin depends on nothing
            subruns = None
        else:
            subruns = _subrunses[0]
        return subruns