from strax import CutPlugin


class EventBuilding(CutPlugin):
    __version__ = "0.0.0"
    depends_on = "event_basics"
    provides = "cut_event_building"
    cut_name = "cut_event_building"
    cut_description = "Whether the salting S1 or S2 can be the main S1 or S2"

    def cut_by(self, events):
        mask = events["s1_salt_number"] >= 0
        mask &= events["s2_salt_number"] >= 0
        return mask
