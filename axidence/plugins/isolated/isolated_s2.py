from strax import parse_selection, CutPlugin
import straxen


class IsolatedS2Mask(CutPlugin):
    __version__ = "0.0.0"
    depends_on = "event_basics"
    provides = "cut_isolated_s2"
    cut_name = "cut_isolated_s2"
    data_kind = "events"
    cut_description = "Isolated S2 selection, a event-level cut."

    isolated_s2_area_range = straxen.URLConfig(
        default=(1, 150),
        type=(list, tuple),
        help="Range of isolated S2 area",
    )

    isolated_s2_selection = straxen.URLConfig(
        default=None,
        type=(str, None),
        help="Selection string for isolated S2",
    )

    def cut_by(self, events):
        mask = events["s2_area"] >= self.isolated_s2_area_range[0]
        mask &= events["s2_area"] <= self.isolated_s2_area_range[1]
        if self.isolated_s2_selection is not None:
            mask &= parse_selection(events, self.isolated_s2_selection)
        return mask
