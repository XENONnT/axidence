import pytest
from unittest import TestCase
from straxen.test_utils import nt_test_context, nt_test_run_id


@pytest.mark.usefixtures("rm_strax_data")
class TestSalting(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_id = nt_test_run_id
        cls.st = nt_test_context()
        cls.st.salt_to_context()

    def test_salting(self):
        """Test the computing of salting plugins."""
        peak_level_plugins = [
            "peaks_salted",
            "peak_proximity_salted",
            "peak_shadow_salted",
            "peak_ambience_salted",
            # "peak_nearest_triggering_salted",
            # "peak_se_score_salted",
        ]
        event_level_plugins = [
            "events_salted",
            "event_basics_salted",
            "event_shadow_salted",
            "event_ambience_salted",
            # "event_nearest_triggering_salted",
            # "event_se_score_salted",
            "events_combine",
            "cuts_event_building_salted",
        ]
        self.st.make(self.run_id, "run_meta", save="run_meta")
        self.st.make(self.run_id, "events_salting", save="events_salting")
        for p in peak_level_plugins + event_level_plugins:
            self.st.make(self.run_id, p, save=p)
