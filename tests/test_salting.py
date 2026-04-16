from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from unittest import TestCase
from straxen.test_utils import nt_test_context, nt_test_run_id

import axidence  # noqa: F401  -- registers salt_to_context on strax.Context


BOOTSTRAP_CSV = Path(__file__).parent / "data" / "bootstrap_areas.csv"


@pytest.mark.usefixtures("rm_strax_data")
class TestSalting(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_id = nt_test_run_id
        # TODO: xenonnt_online should be used here
        cls.st = nt_test_context("xenonnt")
        cls.st.salt_to_context()

    def test_salting(self):
        """Test the computing of salting plugins."""
        peak_level_plugins = [
            "peaks_salted",
            "peak_proximity_salted",
            "peak_shadow_salted",
            "peak_ambience_salted",
            "peak_nearest_triggering_salted",
        ]
        event_level_plugins = [
            "events_salted",
            "event_basics_salted",
            "event_shadow_salted",
            "event_ambience_salted",
            "event_nearest_triggering_salted",
            "events_combine",
            "cuts_event_building_salted",
        ]
        self.st.make(self.run_id, "run_meta", save="run_meta")
        self.st.make(self.run_id, "events_salting", save="events_salting")
        for p in peak_level_plugins + event_level_plugins:
            self.st.make(self.run_id, p, save=p)


@pytest.mark.usefixtures("rm_strax_data")
class TestBootstrapSalting(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_id = nt_test_run_id
        cls.st = nt_test_context("xenonnt")
        cls.st.set_config(
            {
                "s1_distribution": "bootstrap",
                "s2_distribution": "bootstrap",
                "bootstrap_csv": str(BOOTSTRAP_CSV),
            }
        )
        cls.st.salt_to_context()

    def test_bootstrap_pairs_drawn_from_csv(self):
        """Salted s1_area / s2_area pairs are drawn jointly from the bootstrap
        CSV."""
        self.st.make(self.run_id, "run_meta", save="run_meta")
        self.st.make(self.run_id, "events_salting", save="events_salting")

        events = self.st.get_array(self.run_id, "events_salting")
        assert len(events) > 0, "Expected non-empty salted events from bootstrap mode."

        pool = pd.read_csv(BOOTSTRAP_CSV)
        pool_pairs = set(
            zip(
                pool["s1_area"].astype(np.float32).tolist(),
                pool["s2_area"].astype(np.float32).tolist(),
            )
        )
        sampled_pairs = set(
            zip(
                events["s1_area"].astype(np.float32).tolist(),
                events["s2_area"].astype(np.float32).tolist(),
            )
        )
        unknown = sampled_pairs - pool_pairs
        assert not unknown, (
            f"Found {len(unknown)} salted (s1_area, s2_area) pairs that are not in the "
            f"bootstrap CSV: {list(unknown)[:5]}"
        )


@pytest.mark.usefixtures("rm_strax_data")
class TestBootstrapMisconfig:
    def test_only_one_distribution_set_to_bootstrap_raises(self):
        st = nt_test_context("xenonnt")
        st.set_config(
            {
                "s1_distribution": "bootstrap",
                "bootstrap_csv": str(BOOTSTRAP_CSV),
            }
        )
        st.salt_to_context()
        st.make(nt_test_run_id, "run_meta", save="run_meta")
        with pytest.raises(ValueError, match="BOTH s1_distribution and s2_distribution"):
            st.make(nt_test_run_id, "events_salting")

    def test_missing_bootstrap_csv_raises(self):
        st = nt_test_context("xenonnt")
        st.set_config(
            {
                "s1_distribution": "bootstrap",
                "s2_distribution": "bootstrap",
            }
        )
        st.salt_to_context()
        st.make(nt_test_run_id, "run_meta", save="run_meta")
        with pytest.raises(ValueError, match="bootstrap_csv must be set"):
            st.make(nt_test_run_id, "events_salting")
