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
        cls.st = axidence.unsalted_context(output_folder=cls.tempdir)
        cls.st.salt_to_context()

    @classmethod
    def tearDownClass(cls):
        # Make sure to only cleanup this dir after we have done all the tests
        if os.path.exists(cls.tempdir):
            shutil.rmtree(cls.tempdir)

    def test_replication_tree(self):
        """Test the replication_tree method."""
        self.st.replication_tree()
        with self.assertRaises(
            ValueError, msg="Should raise error calling replication_tree twice!"
        ):
            self.st.replication_tree()
