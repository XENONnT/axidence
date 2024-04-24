import axidence


def test_deps_tree():
    """This test can tell us whether the dependency tree is correct or not."""
    st = axidence.unsalted_context()
    st.salt_to_context()
    graph_dir = "./graphs_nT"
    for p in ["event_shadow", "event_ambience", "event_se_density", "cut_event_building"]:
        st.dependency_tree(p, to_dir=graph_dir)
