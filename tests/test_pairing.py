import json
from bson import json_util
from unittest import TestCase
import pandas as pd
import strax
from straxen.test_utils import nt_test_context, nt_test_run_id


def _write_run_doc(context, run_id, storage, start, end):
    """Function which writes a dummy run document."""
    time = pd.to_datetime(start, unit="ns", utc=True)
    endtime = pd.to_datetime(end, unit="ns", utc=True)
    run_doc = {"name": run_id, "start": time, "end": endtime}
    run_doc["comments"] = [{"comment": (endtime - time).total_seconds()}]
    with open(storage._run_meta_path(str(run_id)), "w") as fp:
        json.dump(run_doc, fp, sort_keys=True, indent=4, default=json_util.default)


class TestPairing(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.run_id = nt_test_run_id
        cls.st = nt_test_context()
        cls.st.set_context_config({"write_superruns": True})
        cls.st.salt_and_pair_to_context()

    def test_pairing(self):
        """Test the computing of pairing plugins."""
        hyperrun_name = "__" + self.run_id
        subrun_ids = [self.run_id]
        data_type = "event_basics"
        self.st.make(self.run_id, data_type, save=data_type)
        meta = self.st.get_meta(self.run_id, data_type)
        self.st.storage[0] = strax.DataDirectory(self.st.storage[0].path, provide_run_metadata=True)
        _write_run_doc(
            self.st,
            self.run_id,
            self.st.storage[0],
            meta["start"],
            meta["end"],
        )
        self.st.define_run(hyperrun_name, subrun_ids)
        self.st.check_hyperrun()
        plugins = [
            "peaks_paired",
            "event_info_paired",
            "cut_pairing_exists",
        ]
        for p in plugins:
            self.st.make(self.run_id, p, save=p)
