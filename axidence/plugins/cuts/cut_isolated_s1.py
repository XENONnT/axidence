from strax import parse_selection, CutPlugin
from strax import SaveWhen
import straxen


class IsolatedS1Mask(CutPlugin):
    __version__ = "0.0.0"
    depends_on = "peak_basics"
    provides = "cut_isolated_s1"
    cut_name = "cut_isolated_s1"
    data_kind = "peaks"
    cut_description = "Isolated S1 selection, a peak-level cut."
    save_when = SaveWhen.NEVER

    isolated_s1_area_range = straxen.URLConfig(
        default=(1, 150),
        type=(list, tuple),
        help="Range of isolated S1 area",
    )

    isolated_s1_selection = straxen.URLConfig(
        default=None,
        type=(str, None),
        help="Selection string for isolated S1",
    )

    def cut_by(self, peaks):
        mask = peaks["type"] == 1
        mask &= peaks["area"] >= self.isolated_s1_area_range[0]
        mask &= peaks["area"] <= self.isolated_s1_area_range[1]
        if self.isolated_s1_selection is not None:
            mask &= parse_selection(peaks, self.isolated_s1_selection)
        return mask
