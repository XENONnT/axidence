import pytest
import axidence

    
class TestContext:
    def test_replication_tree(self):
        """Test the replication_tree method."""
        self.st = axidence.ordinary_context()
        self.st.replication_tree()
        with pytest.raises(ValueError):
            self.st.replication_tree()

    @pytest.mark.parametrize("veto_aware", [False, True])
    def test_salt_and_pair_to_context(self, veto_aware):
        """Test the salt_and_pair_to_context method."""
        self.st = axidence.ordinary_context()
        self.st.salt_and_pair_to_context(veto_aware=veto_aware)

        graph_dir = "./graphs_nT"
        self.st.dependency_tree("event_info", to_dir=graph_dir)
        self.st.dependency_tree("event_info_salted", to_dir=graph_dir)
        self.st.dependency_tree("event_info_paired", to_dir=graph_dir)
