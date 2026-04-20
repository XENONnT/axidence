import numpy as np
import strax
import straxen


# `straxen.misc.kind_colors` only exists in straxen >= 2.x. The colours are
# purely cosmetic (used for `dependency_tree` graphs); on older straxen we
# silently skip the registration.
try:
    from straxen.misc import kind_colors
except ImportError:  # straxen 1.7.x
    kind_colors = {}

kind_colors.update(
    {
        "run_meta": "#ffff00",
        "events_salting": "#0080ff",
        "peaks_salted": "#00c0ff",
        "events_salted": "#00ffff",
        "peaks_paired": "#ff00ff",
        "truth_paired": "#ff00ff",
        "events_paired": "#ffccff",
        "isolated_s1": "#80ff00",
        "isolated_s2": "#80ff00",
    }
)


def _dtype_discovery_context():
    """A fully-wired xenonnt context used only to look up canonical plugin
    dtypes.

    In straxen 2.2.x the shared `common_config`/`xnt_common_config` does not
    auto-register a DAQ reader, so a bare Context built from it cannot resolve
    `peaks` all the way back to `raw_records`. `xenonnt_online` does, and it
    works on straxen 3.x too, so we use it in both SR1 and main.
    """
    return straxen.contexts.xenonnt_online(_database_init=False)


def peaks_dtype():
    """Canonical per-peak dtype.

    In modern straxen (3.x) the `Peaks` plugin already covers all per-peak
    fields downstream salting/pairing needs (including `center_time`,
    `area_fraction_top`, etc.). In SR1 (straxen 2.2.7) those live on the
    separate `PeakBasics` plugin, so we merge `peaks ∪ peak_basics` to get a
    superset that's compatible with both stacks.
    """
    st = _dtype_discovery_context()
    plugins = st._get_plugins(("peaks", "peak_basics"), "0")
    merged = strax.merged_dtype([plugins["peaks"].dtype, plugins["peak_basics"].dtype])
    # SR1 strax.merged_dtype returns a descriptor list, modern returns an
    # np.dtype. Normalize before unpacking.
    return strax.unpack_dtype(np.dtype(merged))


def peak_positions_dtype():
    st = _dtype_discovery_context()
    data_name = "peak_positions"
    PeakPositionsPlugin0 = st._get_plugins((data_name,), "0")[data_name]
    return strax.unpack_dtype(PeakPositionsPlugin0.dtype)


shadow_fields = [
    "shadow_s2_time_shadow",
    "dt_s2_time_shadow",
    "x_s2_time_shadow",
    "y_s2_time_shadow",
    "dt_s2_position_shadow",
    "shadow_s2_position_shadow",
    "x_s2_position_shadow",
    "y_s2_position_shadow",
    "pdf_s2_position_shadow",
    # `nearest_s1` / `nearest_s2` (the area-of-nearest-large-peak fields) were
    # added to PeakShadow after SR1, so they're omitted in the SR1 release.
    "nearest_dt_s1",
    "nearest_dt_s2",
]

ambience_fields = [
    "n_lh_before",
    "n_s0_before",
    "n_s1_before",
    "n_s2_before",
    "n_s2_near",
    # SR1 PeakAmbience emits per-channel `s_*_before` / `s_s2_near` rather than
    # the consolidated `s_before` field that was introduced later.
    "s_lh_before",
    "s_s0_before",
    "s_s1_before",
    "s_s2_before",
    "s_s2_near",
]

# SR0 release: PeakNearestTriggering doesn't exist in straxen 1.7.x, so the
# triggering peak fields are dropped entirely.
nearest_triggering_fields: list = []

peak_misc_fields = [
    # `proximity_score` was added to PeakProximity after SR1; the SR1 release
    # uses only the n_competing counts.
    "n_competing_left",
    "n_competing",
]

correlation_fields = shadow_fields + ambience_fields + nearest_triggering_fields + peak_misc_fields

event_level_fields = [
    "n_peaks",
    "s1_center_time",
    "s2_center_time",
    "s1_area",
    "alt_s1_area",
    "s2_area",
    "alt_s2_area",
    "s1_index",
    "alt_s1_index",
    "s2_index",
    "alt_s2_index",
    "r",
    "z",
    "r_naive",
    "z_naive",
]
