"""Back-ports of strax symbols that are missing on the SR0 stack (strax 1.2.3,
straxen 1.7.x, Python 3.8).

Each shim falls back to the real strax symbol when it exists, so importing
from `axidence._compat` is safe on modern strax too.

NOTE: the `DownChunkingPlugin` shim is a no-op base class. Modern axidence
combines `ExhaustPlugin + DownChunkingPlugin` to consume all inputs and emit
multiple output chunks; SR0 strax can't stream multiple chunks out of a
single do_compute call. The SR0 release therefore rewrites the two
affected plugins' `compute` methods (EventsSalting, PeaksPaired) to emit a
single concatenated chunk.
"""

import numpy as np
import strax

__all__ = [
    "ExhaustPlugin",
    "DownChunkingPlugin",
    "CutList",
    "parse_selection",
    "set_nan_defaults",
    "get_accumulated_bool",
]


if hasattr(strax, "set_nan_defaults"):
    set_nan_defaults = strax.set_nan_defaults
else:

    def set_nan_defaults(arr):
        """Fill a structured array with NaN (floats) / -1 (ints) defaults."""
        for name in arr.dtype.names:
            kind = arr.dtype[name].kind
            if kind == "f":
                arr[name] = np.nan
            elif kind in ("i", "u"):
                arr[name] = -1


if hasattr(strax, "parse_selection"):
    parse_selection = strax.parse_selection
else:
    import numexpr

    def parse_selection(x, selection):
        """Back-port of strax.parse_selection."""
        if hasattr(selection, "__call__"):
            return selection(x)
        if isinstance(selection, (list, tuple)):
            selection = " & ".join(f"({s})" for s in selection)
        return numexpr.evaluate(selection, local_dict={fn: x[fn] for fn in x.dtype.names})


if hasattr(strax, "ExhaustPlugin"):
    ExhaustPlugin = strax.ExhaustPlugin
else:

    class ExhaustPlugin(strax.Plugin):  # type: ignore[no-redef]
        """Plugin that exhausts all upstream chunks before computing."""

        def _fetch_chunk(self, d, iters, check_end_not_before=None):
            while super()._fetch_chunk(d, iters, check_end_not_before=check_end_not_before):
                pass
            return False

        def do_compute(self, chunk_i=None, **kwargs):
            if chunk_i != 0:
                raise RuntimeError(
                    f"{self.__class__.__name__} is an ExhaustPlugin. "
                    "It should read all chunks together and process them together."
                )
            return super().do_compute(chunk_i=chunk_i, **kwargs)


if hasattr(strax, "DownChunkingPlugin"):
    DownChunkingPlugin = strax.DownChunkingPlugin
else:

    class DownChunkingPlugin(strax.Plugin):  # type: ignore[no-redef]
        """No-op on SR0 strax: serves only as a marker base class for plugins
        that would otherwise use strax>=1.5 multi-chunk-emission support."""


def get_accumulated_bool(array):
    """Compute the AND of every non-time boolean field of a cut array."""
    fields = [f for f in array.dtype.names if f not in ("time", "endtime")]
    res = np.ones(len(array), np.bool_)
    for field in fields:
        res &= array[field]
    return res


if hasattr(strax, "CutList"):
    CutList = strax.CutList
else:

    class CutList(strax.MergeOnlyPlugin):  # type: ignore[no-redef]
        """Back-port of strax.CutList (strax>=1.4)."""

        __version__ = "0.0.0"
        save_when = strax.SaveWhen.TARGET
        cuts = ()
        _depends_on = ()

        def infer_dtype(self):
            dtype = super().infer_dtype()
            dtype += [
                (
                    (
                        f"Boolean AND of all cuts in {self.accumulated_cuts_string}",
                        self.accumulated_cuts_string,
                    ),
                    np.bool_,
                )
            ]
            return dtype

        def compute(self, **kwargs):
            cuts = super().compute(**kwargs)
            cuts_joint = np.zeros(len(cuts), self.dtype)
            strax.copy_to_buffer(
                cuts,
                cuts_joint,
                f"_copy_cuts_{strax.deterministic_hash(self.depends_on)}",
            )
            cuts_joint[self.accumulated_cuts_string] = get_accumulated_bool(cuts)
            return cuts_joint

        @property  # type: ignore[override]
        def depends_on(self):  # noqa: F811
            if not len(self._depends_on):
                deps = []
                for c in self.cuts:
                    deps.extend(strax.to_str_tuple(c.provides))
                self._depends_on = tuple(deps)
            return self._depends_on

        @depends_on.setter
        def depends_on(self, str_or_tuple):
            self._depends_on = strax.to_str_tuple(str_or_tuple)
