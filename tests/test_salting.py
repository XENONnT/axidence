import os
import uuid
import shutil
import tempfile
from unittest import TestCase
from straxen import units

import axidence


class TestSalting(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Maybe keeping one temp dir is a bit overkill
        temp_folder = uuid.uuid4().hex
        cls.tempdir = os.path.join(tempfile.gettempdir(), temp_folder)
        assert not os.path.exists(cls.tempdir)

        cls.run_id = "0" * 6
        cls.st = axidence.unsalted_context(output_folder=cls.tempdir)
        cls.st.salt_to_context()

        cls.st.set_config(
            {
                "real_run_start": 0,
                "real_run_end": units.s * 10,
            }
        )

    @classmethod
    def tearDownClass(cls):
        # Make sure to only cleanup this dir after we have done all the tests
        if os.path.exists(cls.tempdir):
            shutil.rmtree(cls.tempdir)

    # def test_salting(self):
    #     """Test the computing of salting_events and salting_peaks."""
    #     self.st.make(self.run_id, "salting_events", save="salting_events")
    #     self.st.make(self.run_id, "salting_peaks", save="salting_peaks")
