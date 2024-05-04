import numpy as np
import strax
from strax import OverlapWindowPlugin
import straxen
from straxen import Events, EventBasics

from ...utils import copy_dtype

export, __all__ = strax.exporter()


class EventsForcePaired(OverlapWindowPlugin):
    """Mimicking Events Force manually pairing of isolated S1 & S2 Actually NOT
    used in simulation, but just for debug."""

    depends_on = "peaks_paired"
    provides = "events_paired"
    data_kind = "events_paired"
    save_when = strax.SaveWhen.EXPLICIT
    allow_hyperrun = True

    paring_time_interval = straxen.URLConfig(
        default=int(1e8),
        type=int,
        help="The interval which separates two events S1 [ns]",
    )

    def infer_dtype(self):
        dtype_reference = strax.unpack_dtype(self.deps["peaks_paired"].dtype_for("peaks_paired"))
        required_names = ["time", "endtime", "event_number"]
        dtype = copy_dtype(dtype_reference, required_names)
        return dtype

    def get_window_size(self):
        return 10 * self.paring_time_interval

    def compute(self, peaks_paired):
        peaks_event_number_sorted = np.sort(peaks_paired, order=("event_number", "time"))
        event_number, event_number_index, event_number_count = np.unique(
            peaks_event_number_sorted["event_number"], return_index=True, return_counts=True
        )
        event_number_index = np.append(event_number_index, len(peaks_event_number_sorted))
        result = np.zeros(len(event_number), self.dtype)
        result["time"] = peaks_event_number_sorted["time"][event_number_index[:-1]]
        result["endtime"] = strax.endtime(peaks_event_number_sorted)[event_number_index[1:] - 1]
        result["event_number"] = event_number
        return result


@export
class EventInfosPaired(Events):
    """Superset to EventInfo Besides the features in EventInfo, also store the
    shadow and ambience related features and the origin run_id, group_number,
    time of main S1/2 peaks and the pairing-type."""

    __version__ = "0.0.0"
    depends_on = ("event_info_paired", "peaks_paired")
    provides = "event_infos_paired"
    data_kind = "events_paired"
    save_when = strax.SaveWhen.EXPLICIT
    allow_hyperrun = True

    ambience_fields = straxen.URLConfig(
        default=["lh_before", "s0_before", "s1_before", "s2_before", "s2_near"],
        type=(list, tuple),
        help="Needed ambience related fields",
    )

    alternative_peak_add_fields = straxen.URLConfig(
        default=["se_density"],
        type=(list, tuple),
        help="Fields to store also for alternative peaks",
    )

    @property
    def peak_fields(self):
        required_names = []
        for key in ["s2_time_shadow", "s2_position_shadow"]:
            required_names += [f"shadow_{key}", f"dt_{key}"]
        for ambience in self.ambience_fields:
            required_names += [f"n_{ambience}"]
        required_names += [
            "pdf_s2_position_shadow",
            "nearest_dt_s1",
            "nearest_dt_s2",
            "se_density",
            "left_dtime",
            "right_dtime",
        ]
        required_names += [
            "origin_run_id",
            "origin_group_number",
            "origin_time",
            "origin_endtime",
            "origin_center_time",
        ]
        return required_names

    @property
    def event_fields(self):
        dtype_reference = strax.unpack_dtype(self.deps["peaks_paired"].dtype_for("peaks_paired"))
        peaks_dtype = copy_dtype(dtype_reference, self.peak_fields)
        dtype = []
        for d in peaks_dtype:
            dtype += [
                (("Main S1 " + d[0][0], "s1_" + d[0][1]), d[1]),
                (("Main S2 " + d[0][0], "s2_" + d[0][1]), d[1]),
            ]
            if d[0][1] in self.alternative_peak_add_fields:
                dtype += [
                    (("Alternative S1 " + d[0][0], "alt_s1_" + d[0][1]), d[1]),
                    (("Alternative S2 " + d[0][0], "alt_s2_" + d[0][1]), d[1]),
                ]
        dtype += [
            (
                ("Type of event indicating whether the isolated S1 becomes main S1", "event_type"),
                np.int8,
            ),
            (("Event number in this dataset", "event_number"), np.int64),
            (("Normalization of number of paired events", "normalization"), np.float32),
        ]
        return dtype

    def infer_dtype(self):
        return strax.merged_dtype(
            [
                self.deps["event_info_paired"].dtype_for("event_info_paired"),
                np.dtype(self.event_fields),
            ]
        )

    def compute(self, events_paired, peaks_paired):
        result = np.zeros(len(events_paired), dtype=self.dtype)

        # assign the additional fields
        EventBasics.set_nan_defaults(result)

        # assign the features already in EventInfo
        for q in self.deps["event_info_paired"].dtype_for("event_info_paired").names:
            result[q] = events_paired[q]

        # store AC-type
        split_peaks = strax.split_by_containment(peaks_paired, events_paired)
        for i, (event, sp) in enumerate(zip(events_paired, split_peaks)):
            if np.unique(sp["event_number"]).size != 1:
                raise ValueError(
                    f"Event {i} has multiple event numbers: "
                    f"{np.unique(sp['event_number'])}. "
                    "Maybe the paired events overlap."
                )
            result["event_number"][i] = sp["event_number"][0]
            result["normalization"][i] = sp["normalization"][0]
            for idx, main_peak in zip([event["s1_index"], event["s2_index"]], ["s1_", "s2_"]):
                if idx >= 0:
                    for n in self.peak_fields:
                        result[main_peak + n][i] = sp[n][idx]
            for idx, main_peak in zip(
                [event["alt_s1_index"], event["alt_s2_index"]], ["alt_s1_", "alt_s2_"]
            ):
                if idx >= 0:
                    for n in self.peak_fields:
                        if n in self.alternative_peak_add_fields:
                            result[main_peak + n][i] = sp[n][idx]
            # if the event have S2
            if event["s2_index"] != -1:
                if sp["origin_s1_index"][0] == -1:
                    # if isolated S2 is pure-isolated S2(w/o main S1)
                    if event["s1_index"] != -1:
                        # if successfully paired, considered as AC
                        result["event_type"][i] = 1
                    else:
                        # if unsuccessfully paired, not considered as AC
                        result["event_type"][i] = 2
                else:
                    # if isolated S2 is ext-isolated S2(w/ main S1)
                    if event["s1_index"] != -1 and sp["origin_group_type"][event["s1_index"]] == 1:
                        # if successfully paired and main S1 is from isolated S1 but not isolated S2
                        # considered as AC
                        result["event_type"][i] = 3
                    else:
                        # otherwise, not considered as AC
                        result["event_type"][i] = 4
            else:
                result["event_type"][i] = 5

        result["time"] = events_paired["time"]
        result["endtime"] = events_paired["endtime"]

        return result
