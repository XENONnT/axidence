import os
import uuid
import shutil
import tempfile
from unittest import TestCase

import axidence


class TestContext(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Maybe keeping one temp dir is a bit overkill
        temp_folder = uuid.uuid4().hex
        cls.tempdir = os.path.join(tempfile.gettempdir(), temp_folder)
        assert not os.path.exists(cls.tempdir)

        cls.run_id = "0" * 6

    @classmethod
    def tearDownClass(cls):
        # Make sure to only cleanup this dir after we have done all the tests
        if os.path.exists(cls.tempdir):
            shutil.rmtree(cls.tempdir)

    def setUp(self) -> None:
        self.st = axidence.ordinary_context(output_folder=self.tempdir)

    def test_replication_tree(self):
        """Test the replication_tree method."""
        self.st.replication_tree()
        with self.assertRaises(
            ValueError, msg="Should raise error calling replication_tree twice!"
        ):
            self.st.replication_tree()

    def test_salt_and_pair_to_context(self):
        """Test the salt_and_pair_to_context method."""
        self.st.salt_and_pair_to_context()

        graph_dir = "./graphs_nT"
        self.st.dependency_tree("event_info", to_dir=graph_dir)
        self.st.dependency_tree("event_info_paired", to_dir=graph_dir)
        self.st.dependency_tree("event_info_salted", to_dir=graph_dir)
