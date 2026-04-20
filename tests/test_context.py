from unittest import TestCase

import axidence


class TestContext(TestCase):
    def setUp(self) -> None:
        self.st = axidence.ordinary_context()

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

        # SR0 release: `strax.Context.dependency_tree` was added later; when
        # it's missing, just exercising `salt_and_pair_to_context` is enough
        # to validate the replication-tree registration.
        if not hasattr(self.st, "dependency_tree"):
            return

        graph_dir = "./graphs_nT"
        self.st.dependency_tree("event_info", to_dir=graph_dir)
        self.st.dependency_tree("event_info_salted", to_dir=graph_dir)
        self.st.dependency_tree("event_info_paired", to_dir=graph_dir)
