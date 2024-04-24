import axidence


def test_deps_tree():
    """This test can tell us whether the dependency tree is correct or not."""
    st = axidence.unsalted_context()
    st.salt_to_context()
    st.dependency_tree("event_shadow", to_dir="./graphs_nT")
    st.dependency_tree("event_ambience", to_dir="./graphs_nT")
    st.dependency_tree("event_se_density", to_dir="./graphs_nT")
