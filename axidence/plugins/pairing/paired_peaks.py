import warnings
from immutabledict import immutabledict
import numpy as np
import strax
import straxen
from straxen import units

from ...dtypes import positioned_peak_dtype
from ...plugin import ExhaustPlugin, RunMetaPlugin


class PairedPeaks(ExhaustPlugin, RunMetaPlugin):
    __version__ = "0.0.0"
    depends_on = ("isolated_s1", "isolated_s2")
    provides = ("paired_peaks", "paired_truth")
    data_kind = immutabledict(zip(provides, provides))
    save_when = immutabledict(zip(provides, [strax.SaveWhen.EXPLICIT, strax.SaveWhen.ALWAYS]))

    pairing_seed = straxen.URLConfig(
        default=None,
        type=(int, None),
        help="Seed for pairing",
    )

    isolated_peaks_fields = straxen.URLConfig(
        default=np.dtype(positioned_peak_dtype()).names,
        type=(list, tuple),
        help="Needed fields in isolated peaks",
    )

    isolated_events_fields = straxen.URLConfig(
        default=[],
        type=(list, tuple),
        help="Needed fields in isolated events",
    )

    min_drift_length = straxen.URLConfig(
        default=0,
        type=(int, float),
        help="Manually set minimum length of drifting [cm]",
    )

    max_drift_length = straxen.URLConfig(
        default=straxen.tpc_z,
        infer_type=False,
        help="Total length of the TPC from the bottom of gate to the top of cathode wires [cm]",
    )

    electron_drift_velocity = straxen.URLConfig(
        default="cmt://electron_drift_velocity?version=ONLINE&run_id=plugin.run_id",
        cache=True,
        help="Vertical electron drift velocity in cm/ns (1e4 m/ms)",
    )

    n_drift_time_bin = straxen.URLConfig(
        default=50,
        type=int,
        help="Bin number of drift time",
    )

    isolated_s1_rate_correction = straxen.URLConfig(
        default=True,
        type=bool,
        help="Whether correct isolated S1 rate when calculating AC rate",
    )

    # multiple factor in simulation, e.g. AC rate is 1/t/y,
    # multiple factor is 100, then we will make 100 AC events
    paring_rate_bootstrap_factor = straxen.URLConfig(
        default=1e2,
        type=(int, float),
        help="Bootstrap factor for AC rate",
    )

    apply_shadow_matching = straxen.URLConfig(
        default=True,
        type=bool,
        help="Whether perform shadow matching",
    )

    shift_dt_shadow_matching = straxen.URLConfig(
        default=True,
        type=bool,
        help="Whether shift drift time when performing shadow matching",
    )

    fixed_drift_time = straxen.URLConfig(
        default=None,
        type=(int, float, None),
        help="The fixed drift time [ns]",
    )

    paring_event_interval = straxen.URLConfig(
        default=int(1e8),
        type=int,
        help="The interval which separates two events S1 [ns]",
    )

    def infer_dtype(self):
        dtype = strax.unpack_dtype(self.deps["isolated_s1"].dtype_for("isolated_s1"))
        # TODO: reconsider about how to store run_id after implementing super runs
        peaks_dtype = dtype + [
            (("Event number in this dataset", "event_number"), np.int32),
            # (("Original run id", "origin_run_id"), np.int32),
            (("Original isolated S1/S2 group", "origin_group_number"), np.int32),
            (("Original time of peaks", "origin_time"), np.int64),
            (("Original endtime of peaks", "origin_endtime"), np.int64),
            (("Original center_time of peaks", "origin_center_time"), np.int64),
            (("Original n_competing", "origin_n_competing"), np.int32),
            (("Original type of group", "origin_group_type"), np.int8),
            (("Original s1_index in isolated S1", "origin_s1_index"), np.int32),
            (("Original s2_index in isolated S2", "origin_s2_index"), np.int32),
        ]
        truth_dtype = [
            (("Event number in this dataset", "event_number"), np.int32),
            # (("Original run id of isolated S1", "s1_run_id"), np.int32),
            # (("Original run id of isolated S2", "s2_run_id"), np.int32),
            (
                ("Drift time between isolated S1 and main isolated S2 [ns]", "drift_time"),
                np.float32,
            ),
            (("Original isolated S1 group", "s1_group_number"), np.int32),
            (("Original isolated S2 group", "s2_group_number"), np.int32),
        ] + strax.time_fields
        return dict(paired_peaks=peaks_dtype, paired_truth=truth_dtype)

    def setup(self, prepare=True):
        self.init_run_meta()
        self.min_drift_time = int(self.min_drift_length / self.electron_drift_velocity)
        self.max_drift_time = int(self.max_drift_length / self.electron_drift_velocity)
        if self.pairing_seed is None:
            self.rng = np.random.default_rng(seed=int(self.run_id))
        else:
            self.rng = np.random.default_rng(seed=self.pairing_seed)

    @staticmethod
    def preprocess_isolated_s2(s2):
        # index of isolated S2 groups
        _, s2_group_index, s2_n_peaks = np.unique(
            s2["group_number"], return_index=True, return_counts=True
        )
        # index of main S2s in isolated S2
        s2_main_index = s2_group_index + s2["s2_index"][s2_group_index]

        indices = np.append(s2_group_index, len(s2))
        # time coverage of the first time and last endtime of S2s in each group
        s2_length = s2["endtime"][indices[1:] - 1] - s2["time"][indices[:-1]]
        return s2_group_index, s2_main_index, s2_n_peaks, s2_length

    @staticmethod
    def simple_pairing(
        s1,
        s2,
        run_time,
        max_drift_time,
        min_drift_time,
        paring_rate_correction,
        paring_rate_bootstrap_factor,
        fixed_drift_time,
        rng,
    ):
        s1_rate = len(s1) / run_time
        s2_rate = len(s2) / run_time
        print(f"There are {len(s1)} S1 peaks group")
        print(f"S1 rate is {s1_rate:.3f}Hz")
        print(f"There are {len(s2)} S2 peaks group")
        print(f"S2 rate is {s2_rate * 1e3:.3f}mHz")

        paring_rate_full = (
            s1_rate
            * s2_rate
            * (max_drift_time - min_drift_time)
            / units.s
            * run_time
            / paring_rate_correction
        )
        n_events = round(paring_rate_full * paring_rate_bootstrap_factor)
        s1_group_number = rng.choice(len(s1), size=n_events, replace=True)
        s2_group_number = rng.choice(len(s2), size=n_events, replace=True)
        if fixed_drift_time is None:
            drift_time = rng.uniform(min_drift_time, max_drift_time, size=n_events)
        else:
            warnings.warn(f"Using fixed drift time {fixed_drift_time}ns")
            drift_time = np.full(n_events, fixed_drift_time)
        return paring_rate_full, s1_group_number, s2_group_number, drift_time

    def split_chunks(self, n_peaks):
        # divide results into chunks
        # max peaks number in left_i chunk
        max_in_chunk = round(
            self.chunk_target_size_mb * 1e6 / self.dtype["paired_peaks"].itemsize * 0.9
        )
        _n_peaks = n_peaks.copy()
        if _n_peaks.max() > max_in_chunk:
            raise ValueError("Can not fit a single paired event in a chunk!")
        seen = 0
        slices = [[0, 1]]
        for i in range(len(_n_peaks)):
            if seen + _n_peaks[i] <= max_in_chunk:
                slices[-1][1] = i + 1
                seen += _n_peaks[i]
            else:
                slices += [[i, i + 1]]
                seen = _n_peaks[i]
        return slices

    def build_arrays(
        self,
        drift_time,
        s1_group_number,
        s2_group_number,
        n_peaks,
        s1,
        s2,
        main_s2,
        s2_group_index,
    ):

        # set center time of S1 & S2
        # paired events are separated by roughly `event_interval`
        s1_center_time = (
            np.arange(len(drift_time)).astype(int) * self.paring_event_interval + self.run_start
        )
        s2_center_time = s1_center_time + drift_time
        # total number of isolated S1 & S2 peaks
        peaks_arrays = np.zeros(n_peaks.sum(), dtype=self.dtype["paired_peaks"])

        # assign features of sampled isolated S1 and S2 in AC events
        peaks_count = 0
        for i in range(len(n_peaks)):
            _array = np.zeros(n_peaks[i], dtype=self.dtype["paired_peaks"])
            # isolated S1 is assigned peak by peak
            s1_index = s1_group_number[i]
            for q in self.dtype["paired_peaks"].names:
                if "origin" not in q and q not in ["event_number"]:
                    _array[0][q] = s1[s1_index][q]
            # _array[0]["origin_run_id"] = s1["run_id"][s1_index]
            _array[0]["origin_group_number"] = s1["group_number"][s1_index]
            _array[0]["origin_time"] = s1["time"][s1_index]
            _array[0]["origin_endtime"] = strax.endtime(s1)[s1_index]
            _array[0]["origin_center_time"] = s1["center_time"][s1_index]
            _array[0]["origin_n_competing"] = s1["n_competing"][s1_index]
            _array[0]["origin_group_type"] = 1
            _array[0]["time"] = s1_center_time[i] - (
                s1["center_time"][s1_index] - s1["time"][s1_index]
            )
            _array[0]["endtime"] = _array[0]["time"] + (s1["length"] * s1["dt"])[s1_index]

            # isolated S2 is assigned group by group
            group_number = s2_group_number[i]
            s2_group_i = s2[s2_group_index[group_number] : s2_group_index[group_number + 1]]
            for q in self.dtype["paired_peaks"].names:
                if "origin" not in q and q not in ["event_number"]:
                    _array[1:][q] = s2_group_i[q]
            s2_index = s2_group_i["s2_index"]
            # _array[1:]["origin_run_id"] = s2_group_i["run_id"]
            _array[1:]["origin_group_number"] = s2_group_i["group_number"]
            _array[1:]["origin_time"] = s2_group_i["time"]
            _array[1:]["origin_endtime"] = strax.endtime(s2_group_i)
            _array[1:]["origin_center_time"] = s2_group_i["center_time"]
            _array[1:]["origin_n_competing"] = s2_group_i["n_competing"]
            _array[1:]["origin_group_type"] = 2
            _array[1:]["time"] = s2_center_time[i] - (
                s2_group_i["center_time"][s2_index] - s2_group_i["time"]
            )
            _array[1:]["endtime"] = _array[1:]["time"] + s2_group_i["length"] * s2_group_i["dt"]
            # to avoid overlapping peak(which is really rare case),
            # just move time of input S1, by +1 ns
            while np.any(_array[1:]["time"] == _array[0]["time"]):
                _array[0]["time"] += 1
                _array[0]["endtime"] += 1
                warnings.warn("Isolated S1 overlapped with isolated S2!")

            _array["event_number"] = i
            _array["origin_s1_index"] = s2_group_i["s1_index"][0]
            _array["origin_s2_index"] = s2_index[0]

            peaks_arrays[peaks_count : peaks_count + len(_array)] = _array
            peaks_count += len(_array)

        # assign truth
        truth_arrays = np.zeros(len(n_peaks), dtype=self.dtype["paired_truth"])
        truth_arrays["time"] = peaks_arrays["time"][
            np.unique(peaks_arrays["event_number"], return_index=True)[1]
        ]
        truth_arrays["endtime"] = peaks_arrays["endtime"][
            np.unique(peaks_arrays["event_number"], return_index=True)[1]
        ]
        truth_arrays["event_number"] = np.arange(len(n_peaks))
        truth_arrays["drift_time"] = drift_time
        # truth_arrays["s1_run_id"] = s1["run_id"][s1_group_number]
        # truth_arrays["s2_run_id"] = main_s2["run_id"][s2_group_number]
        truth_arrays["s1_group_number"] = s1["group_number"][s1_group_number]
        truth_arrays["s2_group_number"] = main_s2["group_number"][s2_group_number]

        event_number_index = np.unique(
            peaks_arrays["event_number"], return_index=True, return_counts=True
        )[1]
        if np.any(
            peaks_arrays["time"][event_number_index[1:]]
            <= peaks_arrays["endtime"][event_number_index[1:] - 1]
        ):
            raise ValueError("Some paired events overlap!")

        peaks_arrays = np.sort(peaks_arrays, order=("time", "event_number"))

        if peaks_count != len(peaks_arrays):
            raise ValueError(
                "Mismatch in total number of peaks in the chunk, "
                f"expected {peaks_count}, got {len(peaks_arrays)}!"
            )

        # check overlap of peaks
        n_overlap = (peaks_arrays["time"][1:] - peaks_arrays["endtime"][:-1] < 0).sum()
        if n_overlap:
            warnings.warn(f"{n_overlap} peaks overlap")

        return peaks_arrays, truth_arrays

    def compute(self, isolated_s1, isolated_s2):
        for i, s in enumerate([isolated_s1, isolated_s2]):
            if np.any(np.diff(s["group_number"]) < 0):
                raise ValueError(f"Group number is not sorted in isolated S{i}!")
        s2_group_index, s2_main_index, s2_n_peaks, s2_length = self.preprocess_isolated_s2(
            isolated_s2
        )
        # main S2s in isolated S2
        main_isolated_s2 = isolated_s2[s2_main_index]

        if self.isolated_s1_rate_correction:
            raise NotImplementedError("AC rate correction for isolated S1 is not implemented yet!")
        else:
            paring_rate_correction = 1
        print(f"Input S1 correction factor is {paring_rate_correction:.3f}")

        if self.apply_shadow_matching:
            # simulate AC's drift time bin by bin
            # print(f"Drift time bins number is {self.n_drift_time_bin}")
            # drift_time_bins = np.linspace(
            #     self.min_drift_time, self.max_drift_time, self.n_drift_time_bin + 1
            # )
            # drift_time_bin_center = (drift_time_bins[:-1] + drift_time_bins[1:]) / 2
            raise NotImplementedError("Shadow matching is not implemented yet!")
        else:
            paring_rate_full, s1_group_number, s2_group_number, drift_time = self.simple_pairing(
                isolated_s1,
                main_isolated_s2,
                self.run_time,
                self.max_drift_time,
                self.min_drift_time,
                paring_rate_correction,
                self.paring_rate_bootstrap_factor,
                self.fixed_drift_time,
                self.rng,
            )

        print(f"AC pairing rate is {paring_rate_full * 1e3:.3f}mHz")
        print(f"AC event number is {len(drift_time)}")

        # make sure events are not very long
        assert (s2_length.max() + drift_time.max()) * 5.0 < self.paring_event_interval

        # peaks number in each event
        n_peaks = 1 + s2_n_peaks[s2_group_number]
        slices = self.split_chunks(n_peaks)

        if len(slices) > 1:
            raise NotImplementedError(
                f"Got {len(slices)} chunks. Multiple chunks are not implemented yet!"
            )

        chunk_i = 0
        left_i, right_i = slices[chunk_i]
        peaks_arrays, truth_arrays = self.build_arrays(
            drift_time[left_i:right_i],
            s1_group_number[left_i:right_i],
            s2_group_number[left_i:right_i],
            n_peaks[left_i:right_i],
            isolated_s1,
            isolated_s2,
            main_isolated_s2,
            s2_group_index,
        )
        peaks_arrays["event_number"] += left_i
        truth_arrays["event_number"] += left_i

        start = (
            self.run_start + left_i * self.paring_event_interval - self.paring_event_interval // 2
        )
        end = (
            self.run_start + right_i * self.paring_event_interval - self.paring_event_interval // 2
        )
        result = dict()
        result["paired_peaks"] = self.chunk(
            start=start, end=end, data=peaks_arrays, data_type="paired_peaks"
        )
        result["paired_truth"] = self.chunk(
            start=start, end=end, data=truth_arrays, data_type="paired_truth"
        )
        # chunk size should be less than default chunk size in strax
        assert result["paired_peaks"].nbytes < self.chunk_target_size_mb * 1e6

        return result
