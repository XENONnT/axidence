import numpy as np
import strax
import straxen


def grouped_peak_dtype(n_channels=straxen.n_tpc_pmts):
    dtype = strax.peak_dtype(n_channels=n_channels)
    # since event_number is int64 in event_basics
    dtype += [
        (("Group number of peaks", "group_number"), np.int64),
    ]
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

correlation_fields = shadow_fields + ambience_fields + se_density_fields + nearest_triggering_fields

event_level_fields = [
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
