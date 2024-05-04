import numpy as np
from strax import CutPlugin


class PairingExists(CutPlugin):
    """Cut of successfully paired AC with AC-type(`event_type`)"""

    __version__ = "0.0.0"
    depends_on = "event_infos_paired"
    provides = "cut_pairing_exists"
    cut_name = "cut_pairing_exists"
    data_kind = "events_paired"
    cut_description = (
        "Whether isolated S2 influenced by pairing, and whether the event is considered as AC event"
    )
    allow_hyperrun = True

    def cut_by(self, events_paired):
        return np.isin(events_paired["event_type"], [1, 3])
