import json
from bson import json_util
import pytest
from unittest import TestCase
import pandas as pd
import strax
from straxen.test_utils import nt_test_context, nt_test_run_id

import axidence  # noqa: F401  -- registers salt_and_pair_to_context on strax.Context


def _write_run_doc(context, run_id, storage, start, end):
    """Function which writes a dummy run document."""
    time = pd.to_datetime(start, unit="ns", utc=True)
    endtime = pd.to_datetime(end, unit="ns", utc=True)
    run_doc = {"name": run_id, "start": time, "end": endtime}
    run_doc["comments"] = [{"comment": (endtime - time).total_seconds()}]
    with open(storage._run_meta_path(str(run_id)), "w") as fp:
        json.dump(run_doc, fp, sort_keys=True, indent=4, default=json_util.default)


@pytest.mark.usefixtures("rm_strax_data")
class TestPairing(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_id = nt_test_run_id
        # TODO: xenonnt_online should be used here
        cls.st = nt_test_context("xenonnt")
        cls.st.set_context_config({"write_superruns": True})
        cls.st.salt_and_pair_to_context()

    def test_pairing(self):
        """Test the computing of pairing plugins on a superrun.

        strax 1.2.3 (SR0) has no hyperrun support: a superrun target that is not stored yet is
        made subrun by subrun and the pieces are concatenated, so pairing is per run. Making the
        paired plugins for the superrun therefore also has to produce the subrun copies.
        """
        superrun_name = "_" + self.run_id
        subrun_ids = [self.run_id]
        data_type = "event_basics"
        self.st.make(self.run_id, data_type, save=data_type)
        meta = self.st.get_metadata(self.run_id, data_type)
        self.st.storage[0] = strax.DataDirectory(self.st.storage[0].path, provide_run_metadata=True)
        _write_run_doc(
            self.st,
            self.run_id,
            self.st.storage[0],
            meta["start"],
            meta["end"],
        )
        self.st.define_run(superrun_name, subrun_ids)
        # `check_superrun` / `check_hyperrun` were added in later strax versions.
        if hasattr(self.st, "check_superrun"):
            self.st.check_superrun()
        plugins = [
            "peaks_paired",
            "event_info_paired",
            "cut_pairing_exists",
        ]
        for p in plugins:
            # a single-subrun superrun relies on the `is_stored` shim in axidence._compat
            self.st.make(superrun_name, p, save=p)
            assert self.st.is_stored(superrun_name, p)
            assert self.st.is_stored(self.run_id, p), "strax should have made the subrun first"
            assert len(self.st.get_array(superrun_name, p)) == len(
                self.st.get_array(self.run_id, p)
            )
