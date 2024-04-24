import straxen
from axidence import SaltingEvents, SaltingPeaks
from axidence import (
    SaltingPeakProximity,
    SaltingPeakShadow,
    SaltingPeakAmbience,
    SaltingPeakSEDensity,
)
from axidence import (
    SaltedEvents,
    SaltedEventBasics,
    SaltedEventShadow,
    SaltedEventAmbience,
    SaltedEventSEDensity,
)


def test_deps_tree():
    st = straxen.contexts.xenonnt_online(_database_init=False)
    st.register(
        (
            SaltingEvents,
            SaltingPeaks,
            SaltingPeakProximity,
            SaltingPeakShadow,
            SaltingPeakAmbience,
            SaltingPeakSEDensity,
            SaltedEvents,
            SaltedEventBasics,
            SaltedEventShadow,
            SaltedEventAmbience,
            SaltedEventSEDensity,
        )
    )
    st.dependency_tree("event_shadow", to_dir="./graphs_nT")
    st.dependency_tree("event_ambience", to_dir="./graphs_nT")
    st.dependency_tree("event_se_density", to_dir="./graphs_nT")
