import numpy as np
import strax
import straxen

from straxen.misc import kind_colors


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


def peak_positions_dtype():
    st = strax.Context(
        config=straxen.contexts.xnt_common_config, **straxen.contexts.xnt_common_opts
    )
    data_name = "peak_positions"
    PeakPositionsNT0 = st._get_plugins((data_name,), "0")[data_name]
    return strax.unpack_dtype(PeakPositionsNT0.dtype)


def positioned_peak_dtype(n_channels=straxen.n_tpc_pmts):
    dtype = strax.merged_dtype(
        [
            np.dtype(strax.peak_dtype(n_channels=straxen.n_tpc_pmts)),
            np.dtype(peak_positions_dtype()),
        ]
    )
    return dtype


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
    "nearest_dt_s1",
    "nearest_dt_s2",
]

ambience_fields = ["n_lh_before", "n_s0_before", "n_s1_before", "n_s2_before", "n_s2_near"]

se_density_fields = ["se_density"]

nearest_triggering_fields = []
for direction in ["left", "right"]:
    nearest_triggering_fields += [
        f"{direction}_dtime",
        f"{direction}_time",
        f"{direction}_endtime",
        f"{direction}_type",
        f"{direction}_n_competing",
        f"{direction}_area",
    ]

peak_misc_fields = [
    "center_time",
    "area_fraction_top",
    "n_competing",
    "n_competing_left",
]

correlation_fields = (
    shadow_fields
    + ambience_fields
    + se_density_fields
    + nearest_triggering_fields
    + peak_misc_fields
)

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
