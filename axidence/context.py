import strax
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


def unsalted_context(**kwargs):
    return straxen.contexts.xenonnt_online(_database_init=False, **kwargs)


@strax.Context.add_method
def salt_to_context(self):
    self.register(
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
