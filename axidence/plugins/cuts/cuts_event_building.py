from strax import CutPlugin, CutList


class MainS1Trigger(CutPlugin):
    __version__ = "0.1.0"
    depends_on = "event_basics"
    provides = "cut_main_s1_trigger"
    cut_name = "cut_main_s1_trigger"
    cut_description = "Whether the salting S1 can be the main S2"

    def cut_by(self, events):
        mask = events["s1_salt_number"] == events["salt_number"]
        return mask


class MainS2Trigger(CutPlugin):
    __version__ = "0.1.0"
    depends_on = "event_basics"
    provides = "cut_main_s2_trigger"
    cut_name = "cut_main_s2_trigger"
    cut_description = "Whether the salting S2 can be the main S2"

    def cut_by(self, events):
        mask = events["s2_salt_number"] == events["salt_number"]
        return mask


class EventBuilding(CutList):
    __version__ = "0.0.0"
    provides = "cuts_event_building"
    accumulated_cuts_string = "cuts_event_building"
    cut_description = "Whether the salting S1 or S2 can be the main S1 or S2"

    cuts = (MainS1Trigger, MainS2Trigger)
