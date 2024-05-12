import warnings
from immutabledict import immutabledict
import numpy as np
import strax
from strax import Plugin, ExhaustPlugin, DownChunkingPlugin
import straxen
from straxen import units
from straxen import PeakProximity
import GOFevaluation as ge

from ...utils import copy_dtype
from ...dtypes import peak_positions_dtype


class PeaksPaired(ExhaustPlugin, DownChunkingPlugin):
    __version__ = "0.0.1"
    depends_on = (
        "run_meta",
        "isolated_s1",
        "isolated_s2",
        "peaks_salted",
        "peak_shadow_salted",
    )
    provides = ("peaks_paired", "truth_paired")
    data_kind = immutabledict(zip(provides, provides))
    save_when = immutabledict(zip(provides, [strax.SaveWhen.EXPLICIT, strax.SaveWhen.ALWAYS]))
    rechunk_on_save = immutabledict(zip(provides, [False, True]))
    allow_hyperrun = True

    pairing_seed = straxen.URLConfig(
        default=None,
        type=(int, None),
        help="Seed for pairing",
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

    n_drift_time_bins = straxen.URLConfig(
        default=50,
        type=int,
        help="Bin number of drift time",
    )

    max_n_shadow_bins = straxen.URLConfig(
        default=30,
        type=int,
        help="Max bin number of 2D shadow matching",
    )

    # multiple factor in simulation,
    # e.g. if number of event is is 1 in the run, and multiple factor is 100,
    # then we will make 100 events
    paring_rate_bootstrap_factor = straxen.URLConfig(
        default=1e2,
        type=(int, float, list, tuple),
        help=(
            "Bootstrap factor for pairing rate, "
            "if list or tuple, they are the factor for 2 and 3+ hits S1"
        ),
    )

    s1_min_coincidence = straxen.URLConfig(
        default=2, type=int, help="Minimum tight coincidence necessary to make an S1"
    )

    s2_area_range = straxen.URLConfig(
        default=(1e2, 2e4),
        type=(list, tuple),
        help="Range of S2 area in salting",
    )

    s2_distribution = straxen.URLConfig(
        default="exponential",
        type=str,
        help="S2 distribution shape in salting",
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

    only_salt_s1 = straxen.URLConfig(
        default=False,
        type=bool,
        help="Whether only salt S1",
    )

    shadow_deltatime_exponent = straxen.URLConfig(
        default=-1.0, type=float, track=True, help="The exponent of delta t when calculating shadow"
    )

    fixed_drift_time = straxen.URLConfig(
        default=None,
        type=(int, float, None),
        help="The fixed drift time [ns]",
    )

    paring_time_interval = straxen.URLConfig(
        default=int(1e8),
        type=int,
        help="The interval which separates two events S1 [ns]",
    )

    def infer_dtype(self):
        dtype = strax.unpack_dtype(self.deps["isolated_s1"].dtype_for("isolated_s1"))
        peaks_dtype = dtype + [
            # since event_number is int64 in event_basics
            (("Event number in this dataset", "event_number"), np.int64),
            (("Original run id", "origin_run_id"), np.int32),
            (("Original isolated S1/S2 group", "origin_group_number"), np.int32),
            (("Original time of peaks", "origin_time"), np.int64),
            (("Original endtime of peaks", "origin_endtime"), np.int64),
            (("Original center_time of peaks", "origin_center_time"), np.int64),
            (("Original n_competing", "origin_n_competing"), np.int32),
            (("Original type of group", "origin_group_type"), np.int8),
            (("Original s1_index in isolated S1", "origin_s1_index"), np.int32),
            (("Original s2_index in isolated S2", "origin_s2_index"), np.int32),
            (("Normalization of number of paired events", "normalization"), np.float32),
        ]
        truth_dtype = [
            (("Event number in this dataset", "event_number"), np.int64),
            (("Original run id of isolated S1", "s1_run_id"), np.int32),
            (("Original run id of isolated S2", "s2_run_id"), np.int32),
            (
                ("Drift time between isolated S1 and main isolated S2 [ns]", "drift_time"),
                np.float32,
            ),
            (("Original isolated S1 group", "s1_group_number"), np.int32),
            (("Original isolated S2 group", "s2_group_number"), np.int32),
            (("Normalization of number of paired events", "normalization"), np.float32),
        ] + strax.time_fields
        return dict(peaks_paired=peaks_dtype, truth_paired=truth_dtype)

    def setup(self, prepare=True):
        self.min_drift_time = int(self.min_drift_length / self.electron_drift_velocity)
        self.max_drift_time = int(self.max_drift_length / self.electron_drift_velocity)
        if self.pairing_seed is None:
            self.rng = np.random.default_rng(seed=int(self.run_id))
        else:
            self.rng = np.random.default_rng(seed=self.pairing_seed)
        self.time_left = self.paring_time_interval // 2
        self.time_right = self.paring_time_interval - self.time_left
        if self.s1_min_coincidence != 2:
            raise NotImplementedError("Only support s1_min_coincidence = 2 now!")
        if isinstance(self.paring_rate_bootstrap_factor, (list, tuple)):
            if len(self.paring_rate_bootstrap_factor) != 2:
                raise ValueError(
                    "The length of paring_rate_bootstrap_factor should be 2 "
                    "if provided list or tuple!"
                )
            self.bootstrap_factor = list(self.paring_rate_bootstrap_factor)
        else:
            self.bootstrap_factor = [self.paring_rate_bootstrap_factor] * 2

    @staticmethod
    def update_group_number(isolated, run_meta):
        result = isolated.copy()
        windows = strax.touching_windows(isolated, run_meta)
        n_groups = np.array(
            [np.unique(isolated["group_number"][w[0] : w[1]]).size for w in windows]
        )
        cumsum = np.cumsum(np.hstack([[0], n_groups])[:-1])
        result["group_number"] += np.repeat(cumsum, windows[:, 1] - windows[:, 0])
        return result

    @staticmethod
    def preprocess_isolated_s2(s2):
        # index of isolated S2 groups
        _, s2_group_index, s2_n_peaks = np.unique(
            s2["group_number"], return_index=True, return_counts=True
        )
        # index of main S2s in isolated S2
        s2_main_index = s2_group_index + s2["s2_index"][s2_group_index]

        s2_group_index = np.append(s2_group_index, len(s2))
        # time coverage of the first time and last endtime of S2s in each group
        s2_length = s2["endtime"][s2_group_index[1:] - 1] - s2["time"][s2_group_index[:-1]]
        return s2_group_index, s2_main_index, s2_n_peaks, s2_length

    @staticmethod
    def simple_pairing(
        s1,
        s2,
        s1_rate,
        s2_rate,
        run_time,
        max_drift_time,
        min_drift_time,
        paring_rate_correction,
        paring_rate_bootstrap_factor,
        fixed_drift_time,
        rng,
    ):
        paring_rate_full = (
            s1_rate * s2_rate * (max_drift_time - min_drift_time) / units.s / paring_rate_correction
        )
        n_events = round(paring_rate_full * run_time * paring_rate_bootstrap_factor)
        s1_group_index = rng.choice(len(s1), size=n_events, replace=True)
        s2_group_index = rng.choice(len(s2), size=n_events, replace=True)
        if fixed_drift_time is None:
            drift_time = rng.uniform(min_drift_time, max_drift_time, size=n_events)
        else:
            warnings.warn(f"Using fixed drift time {fixed_drift_time}ns")
            drift_time = np.full(n_events, fixed_drift_time)
        return paring_rate_full, (
            s1["group_number"][s1_group_index],
            s2["group_number"][s2_group_index],
            drift_time,
        )

    def get_paring_rate_correction(self, peaks_salted):
        # need to use peak-level salting because
        # main S1 of event-level salting might be contaminated
        return 1

    def shadow_reference_selection(self, peaks_salted):
        """Select the reference events for shadow matching, also return
        weights."""

        if self.only_salt_s1:
            raise ValueError("Cannot only salt S1 when performing shadow matching!")

        reference = peaks_salted[peaks_salted["type"] == 2]
        weight = np.ones(len(reference))
        weight /= weight.sum()

        dtype = np.dtype(
            [
                ("area", float),
                ("dt_s2_time_shadow", float),
                ("shadow_s2_time_shadow", float),
                ("weight", float),
            ]
        )
        shadow_reference = np.zeros(len(reference), dtype=dtype)
        for n in dtype.names:
            if n == "weight":
                shadow_reference[n] = weight
            else:
                shadow_reference[n] = reference[n]
        return shadow_reference, ""

    @staticmethod
    def digitize2d(data_sample, bin_edges, n_bins):
        """Get indices of the 2D bins to which each value in input array
        belongs.

        Args:
            data_sample: array, data waiting for binning
        """
        digit = np.zeros(len(data_sample), dtype=int)
        # `x_dig` is within [0, len(bin_edges[0])-1]
        x_dig = np.digitize(data_sample[:, 0], bin_edges[0][1:])
        for xd in np.unique(x_dig):
            digit[x_dig == xd] = (
                np.digitize(data_sample[:, 1][x_dig == xd], bin_edges[1][xd][1:]) + xd * n_bins
            )
        return digit

    @staticmethod
    def preprocess_shadow(data, shadow_deltatime_exponent, delta_t=0, prefix=""):
        dt_s2_time_shadow = np.clip(data[f"{prefix}dt_s2_time_shadow"] - delta_t, 1, np.inf)
        pre_s2_area = (
            data[f"{prefix}shadow_s2_time_shadow"]
            * data[f"{prefix}dt_s2_time_shadow"] ** -shadow_deltatime_exponent
        )
        x = np.log10(pre_s2_area * dt_s2_time_shadow**shadow_deltatime_exponent)
        y = np.sqrt(np.log10(pre_s2_area) ** 2 + np.log10(dt_s2_time_shadow) ** 2)
        sample = np.stack([x, y]).T
        # sample = np.stack([np.log10(dt_s2_time_shadow), np.log10(pre_s2_area)]).T
        return sample

    @staticmethod
    def shadow_matching(
        s1,
        s2,
        shadow_reference,
        run_time,
        max_drift_time,
        min_drift_time,
        rng,
        preprocess_shadow,
        paring_rate_bootstrap_factor=1e2,
        paring_rate_correction=1.0,
        shift_dt_shadow_matching=True,
        n_drift_time_bins=50,
        shadow_deltatime_exponent=-1.0,
        max_n_shadow_bins=30,
        prefix="",
        onlyrate=False,
    ):
        if paring_rate_bootstrap_factor < 1.0:
            raise ValueError("Bootstrap factor should be larger than 1!")
        if paring_rate_correction > 1.0:
            raise ValueError("Correction factor should be smaller than 1!")
        # perform Shadow matching technique
        # see details in xenon:xenonnt:ac:prediction:summary#fake-plugin_simulation

        # 2D equal binning
        # prepare the 2D space, x is log(S2/dt), y is (log(S2)**2+log(dt)**2)**0.5
        # because these 2 dimension is orthogonal
        sampled_correlation = preprocess_shadow(
            shadow_reference, shadow_deltatime_exponent, prefix=prefix
        )
        s1_sample = preprocess_shadow(s1, shadow_deltatime_exponent)

        # use (x, y) distribution of isolated S1 as reference
        # because it is more intense when shadow(S2/dt) is large
        reference_sample = s1_sample
        ge.check_sample_sanity(reference_sample)
        # largest bin number in x & y dimension
        n_shadow_bins = max_n_shadow_bins
        # find suitable bin number
        # we do not allow binning which provides empty bin(count = 0) for Shadow's (x, y)
        # because it is denominator
        bins_find = False
        while not bins_find:
            try:
                # binning is order [0, 1], important, place do not change this order
                _, bin_edges = ge.equiprobable_histogram(
                    data_sample=reference_sample[
                        reference_sample[:, 0] != reference_sample[:, 0].min()
                    ],
                    reference_sample=reference_sample[
                        reference_sample[:, 0] != reference_sample[:, 0].min()
                    ],
                    n_partitions=[n_shadow_bins, n_shadow_bins],
                )
                ns = ge.apply_irregular_binning(
                    data_sample=sampled_correlation,
                    bin_edges=bin_edges,
                    data_sample_weights=shadow_reference["weight"],
                )
                if np.any(ns <= 0):
                    raise ValueError(
                        f"Weird! Find empty bin when the bin number is {n_shadow_bins}!"
                    )
                bins_find = True
            except:  # noqa
                n_shadow_bins -= 1
            assert n_shadow_bins > 0, "No suitable binning found"
        print(f"Shadow bins number is {n_shadow_bins}")
        # apply the binning
        s1_shadow_count = ge.apply_irregular_binning(data_sample=s1_sample, bin_edges=bin_edges)
        # s2_shadow_count = ge.apply_irregular_binning(data_sample=s2_sample, bin_edges=bin_edges)
        sampled_shadow_count = ge.apply_irregular_binning(
            data_sample=sampled_correlation,
            bin_edges=bin_edges,
            data_sample_weights=shadow_reference["weight"],
        )
        # shadow_ratio = s1_shadow_count / sampled_shadow_count
        shadow_run_time = sampled_shadow_count / sampled_shadow_count.sum() * run_time
        if not onlyrate:
            # get indices of the 2D bins
            s1_digit = PeaksPaired.digitize2d(s1_sample, bin_edges, n_shadow_bins)
            _s1_group_index = np.arange(len(s1))
            s1_group_index_list = [
                _s1_group_index[s1_digit == xd].tolist()
                for xd in range(n_shadow_bins * n_shadow_bins)
            ]

        drift_time_bins = np.linspace(min_drift_time, max_drift_time, n_drift_time_bins + 1)
        drift_time_bin_center = (drift_time_bins[:-1] + drift_time_bins[1:]) / 2

        group_index_list = []
        _paring_rate_full = np.zeros(len(drift_time_bin_center))
        for i in range(len(drift_time_bin_center)):
            if shift_dt_shadow_matching:
                # different drift time between isolated S1 and S2 indicates
                # different shadow matching arrangement
                delta_t = drift_time_bin_center[i]
            else:
                delta_t = 0
            data_sample = preprocess_shadow(s2, shadow_deltatime_exponent, delta_t=delta_t)
            ge.check_sample_sanity(data_sample)
            # apply binning to (x, y)
            s2_shadow_count = ge.apply_irregular_binning(
                data_sample=data_sample, bin_edges=bin_edges
            )
            # conditional rate is pairing rate in each (x, y) bin
            ac_rate_conditional = (
                s1_shadow_count / shadow_run_time * s2_shadow_count / shadow_run_time
            )
            ac_rate_conditional *= (drift_time_bins[i + 1] - drift_time_bins[i]) / units.s
            ac_rate_conditional *= sampled_shadow_count / sampled_shadow_count.sum()
            ac_rate_conditional /= paring_rate_correction
            _paring_rate_full[i] = ac_rate_conditional.sum()
            if not onlyrate:
                # expectation of pairing in each bin in this run
                lam_shadow = ac_rate_conditional * run_time * paring_rate_bootstrap_factor
                count_pairing = rng.poisson(lam=lam_shadow).flatten()
                if count_pairing.max() == 0:
                    count_pairing[lam_shadow.argmax()] = 1
                s2_digit = PeaksPaired.digitize2d(data_sample, bin_edges, n_shadow_bins)
                _s2_group_index = np.arange(len(s2))
                s2_group_index_list = [
                    _s2_group_index[s2_digit == xd].tolist()
                    for xd in range(n_shadow_bins * n_shadow_bins)
                ]
                # random sample isolated S1 and S2's group number
                _s1_group_index = np.hstack(
                    [
                        rng.choice(
                            s1_group_index_list[xd],
                            size=count_pairing[xd],
                        )
                        for xd in range(n_shadow_bins * n_shadow_bins)
                    ]
                )
                _s2_group_index = np.hstack(
                    [
                        rng.choice(
                            s2_group_index_list[xd],
                            size=count_pairing[xd],
                        )
                        for xd in range(n_shadow_bins * n_shadow_bins)
                    ]
                )
                # sample drift time in this bin
                _drift_time = rng.uniform(
                    drift_time_bins[i],
                    drift_time_bins[i + 1],
                    size=count_pairing.sum(),
                )
                group_index_list.append([_s1_group_index, _s2_group_index, _drift_time])
        paring_rate_full = _paring_rate_full.sum()
        if not onlyrate:
            s1_group_index = np.hstack([group[0] for group in group_index_list]).astype(int)
            s2_group_index = np.hstack([group[1] for group in group_index_list]).astype(int)
            drift_time = np.hstack([group[2] for group in group_index_list]).astype(int)
            assert len(s1_group_index) == len(s2_group_index)
        return paring_rate_full, (
            s1["group_number"][s1_group_index],
            s2["group_number"][s2_group_index],
            drift_time,
        )

    def split_chunks(self, n_peaks):
        # divide results into chunks
        # max peaks number in left_i chunk
        max_in_chunk = round(
            0.9 * self.chunk_target_size_mb * 1e6 / self.dtype["peaks_paired"].itemsize
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
        start,
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
        s1_center_time = np.arange(len(drift_time)).astype(int) * self.paring_time_interval + start
        s2_center_time = s1_center_time + drift_time
        # total number of isolated S1 & S2 peaks
        peaks_arrays = np.zeros(n_peaks.sum(), dtype=self.dtype["peaks_paired"])

        # assign features of sampled isolated S1 and S2 in pairing events
        peaks_count = 0
        for i in range(len(n_peaks)):
            _array = np.zeros(n_peaks[i], dtype=self.dtype["peaks_paired"])
            # isolated S1 is assigned peak by peak
            s1_index = s1_group_number[i]
            for q in self.dtype["peaks_paired"].names:
                if "origin" not in q and q not in ["event_number", "normalization"]:
                    _array[0][q] = s1[s1_index][q]
            _array[0]["origin_run_id"] = s1["run_id"][s1_index]
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
            for q in self.dtype["peaks_paired"].names:
                if "origin" not in q and q not in ["event_number", "normalization"]:
                    _array[1:][q] = s2_group_i[q]
            s2_index = s2_group_i["s2_index"]
            _array[1:]["origin_run_id"] = s2_group_i["run_id"]
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

        if peaks_count != len(peaks_arrays):
            raise ValueError(
                "Mismatch in total number of peaks in the chunk, "
                f"expected {peaks_count}, got {len(peaks_arrays)}!"
            )

        # assign truth
        truth_arrays = np.zeros(len(n_peaks), dtype=self.dtype["truth_paired"])
        truth_arrays["time"] = peaks_arrays["time"][
            np.unique(peaks_arrays["event_number"], return_index=True)[1]
        ]
        truth_arrays["endtime"] = peaks_arrays["endtime"][
            np.unique(peaks_arrays["event_number"], return_index=True)[1]
        ]
        truth_arrays["event_number"] = np.arange(len(n_peaks))
        truth_arrays["drift_time"] = drift_time
        truth_arrays["s1_run_id"] = s1["run_id"][s1_group_number]
        truth_arrays["s2_run_id"] = main_s2["run_id"][s2_group_number]
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

        return peaks_arrays, truth_arrays

    def compute(self, run_meta, isolated_s1, isolated_s2, peaks_salted, start, end):
        isolated_s1 = self.update_group_number(isolated_s1, run_meta)
        isolated_s2 = self.update_group_number(isolated_s2, run_meta)
        for i, s in enumerate([isolated_s1, isolated_s2]):
            if np.any(np.diff(s["group_number"]) < 0):
                raise ValueError(f"Group number is not sorted in isolated S{i}!")
        s2_group_index, s2_main_index, s2_n_peaks, s2_length = self.preprocess_isolated_s2(
            isolated_s2
        )
        # main S2s in isolated S2
        main_isolated_s2 = isolated_s2[s2_main_index]

        paring_rate_correction = self.get_paring_rate_correction(peaks_salted)
        print(f"Isolated S1 correction factor is {paring_rate_correction:.3f}")

        run_time = (run_meta["endtime"] - run_meta["time"]).sum() / units.s
        s1_rate = len(isolated_s1) / run_time
        s2_rate = len(main_isolated_s2) / run_time
        print(f"Total run time is {run_time:.2f}s")
        print(f"There are {len(isolated_s1)} S1 peaks group")
        print(f"S1 rate is {s1_rate:.3f}Hz")
        print(f"There are {len(main_isolated_s2)} S2 peaks group")
        print(f"S2 rate is {s2_rate * 1e3:.3f}mHz")
        n_hits_2 = isolated_s1["n_hits"] == 2
        n_hits_masks = [n_hits_2, ~n_hits_2]
        truths = []
        for i, mask in enumerate(n_hits_masks):
            if mask.sum() != 0:
                if self.apply_shadow_matching:
                    # simulate drift time bin by bin
                    shadow_reference, prefix = self.shadow_reference_selection(peaks_salted)
                    truth = self.shadow_matching(
                        isolated_s1[mask],
                        main_isolated_s2,
                        shadow_reference,
                        run_time,
                        self.max_drift_time,
                        self.min_drift_time,
                        self.rng,
                        self.preprocess_shadow,
                        self.bootstrap_factor[i],
                        paring_rate_correction,
                        self.shift_dt_shadow_matching,
                        self.n_drift_time_bins,
                        self.shadow_deltatime_exponent,
                        self.max_n_shadow_bins,
                        prefix,
                    )
                else:
                    truth = self.simple_pairing(
                        isolated_s1[mask],
                        main_isolated_s2,
                        s1_rate,
                        s2_rate,
                        run_time,
                        self.max_drift_time,
                        self.min_drift_time,
                        paring_rate_correction,
                        self.bootstrap_factor[i],
                        self.fixed_drift_time,
                        self.rng,
                    )
            else:
                truth = (
                    0.0,
                    (
                        np.empty(0, dtype=isolated_s1["group_number"].dtype),
                        np.empty(0, dtype=main_isolated_s2["group_number"].dtype),
                        [],
                    ),
                )
            truths.append(truth)
        paring_rate_full = truths[0][0] + truths[1][0]
        s1_group_number = np.hstack([truths[0][1][0], truths[1][1][0]])
        s2_group_number = np.hstack([truths[0][1][1], truths[1][1][1]])
        drift_time = np.hstack([truths[0][1][2], truths[1][1][2]])
        normalization = np.hstack(
            [
                np.full(len(truths[0][1][0]), 1 / self.bootstrap_factor[0]),
                np.full(len(truths[1][1][0]), 1 / self.bootstrap_factor[1]),
            ]
        )

        print(f"Pairing rate is {paring_rate_full * 1e3:.3f}mHz")
        print(f"Event number is {len(drift_time)}")

        # make sure events are not very long
        assert (s2_length.max() + drift_time.max()) * 5.0 < self.paring_time_interval

        # peaks number in each event
        n_peaks = 1 + s2_n_peaks[s2_group_number]
        slices = self.split_chunks(n_peaks)

        print(f"Number of chunks is {len(slices)}")

        for chunk_i in range(len(slices)):
            left_i, right_i = slices[chunk_i]

            _start = start + left_i * self.paring_time_interval
            _end = start + right_i * self.paring_time_interval

            peaks_arrays, truth_arrays = self.build_arrays(
                _start + self.time_left,
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
            peaks_arrays["normalization"] = np.repeat(
                normalization[left_i:right_i],
                n_peaks[left_i:right_i],
            )
            truth_arrays["normalization"] = normalization[left_i:right_i]

            # becareful with all fields assignment after sorting
            peaks_arrays = np.sort(peaks_arrays, order=("time", "event_number"))

            # check overlap of peaks
            n_overlap = (peaks_arrays["time"][1:] - peaks_arrays["endtime"][:-1] < 0).sum()
            if n_overlap:
                warnings.warn(f"{n_overlap} peaks overlap")

            result = dict()
            result["peaks_paired"] = self.chunk(
                start=_start, end=_end, data=peaks_arrays, data_type="peaks_paired"
            )
            result["truth_paired"] = self.chunk(
                start=_start, end=_end, data=truth_arrays, data_type="truth_paired"
            )
            # chunk size should be less than default chunk size in strax
            assert result["peaks_paired"].nbytes < self.chunk_target_size_mb * 1e6

            yield result


class PeakProximityPaired(PeakProximity):
    __version__ = "0.0.0"
    depends_on = "peaks_paired"
    provides = "peak_proximity_paired"
    data_kind = "peaks_paired"
    save_when = strax.SaveWhen.EXPLICIT
    allow_hyperrun = True

    use_origin_n_competing = straxen.URLConfig(
        default=False,
        type=bool,
        help="Whether use original n_competing",
    )

    def infer_dtype(self):
        dtype_reference = strax.unpack_dtype(self.deps["peaks_paired"].dtype_for("peaks_paired"))
        required_names = ["time", "endtime", "n_competing"]
        dtype = copy_dtype(dtype_reference, required_names)
        return dtype

    def compute(self, peaks_paired):
        if self.use_origin_n_competing:
            warnings.warn("Using original n_competing for paired peaks")
            n_competing = peaks_paired["origin_n_competing"].copy()
        else:
            # add `n_competing` to isolated S1 and isolated S2 because injection of peaks
            # will not consider the competing window because
            # that window is much larger than the max drift time
            n_competing = np.zeros(len(peaks_paired), self.dtype["n_competing"])
            peaks_event_number_sorted = np.sort(peaks_paired, order=("event_number", "time"))
            event_number, event_number_index, event_number_count = np.unique(
                peaks_event_number_sorted["event_number"],
                return_index=True,
                return_counts=True,
            )
            event_number_index = np.append(event_number_index, len(peaks_event_number_sorted))
            for i in range(len(event_number)):
                areas = peaks_event_number_sorted["area"][
                    event_number_index[i] : event_number_index[i + 1]
                ].copy()
                types = peaks_event_number_sorted["origin_group_type"][
                    event_number_index[i] : event_number_index[i + 1]
                ].copy()
                n_competing_s = peaks_event_number_sorted["origin_n_competing"][
                    event_number_index[i] : event_number_index[i + 1]
                ].copy()
                threshold = areas * self.min_area_fraction
                for j in range(event_number_count[i]):
                    if types[j] == 1:
                        n_competing_s[j] += np.sum(areas[types == 2] > threshold[j])
                    elif types[j] == 2:
                        n_competing_s[j] += np.sum(areas[types == 1] > threshold[j])
                n_competing[event_number_index[i] : event_number_index[i + 1]] = n_competing_s

        return dict(
            time=peaks_paired["time"],
            endtime=strax.endtime(peaks_paired),
            n_competing=n_competing[peaks_event_number_sorted["time"].argsort()],
        )


class PeakPositionsPaired(Plugin):
    __version__ = "0.0.0"
    depends_on = "peaks_paired"
    provides = "peak_positions_paired"
    save_when = strax.SaveWhen.EXPLICIT
    allow_hyperrun = True

    def infer_dtype(self):
        return peak_positions_dtype()

    def compute(self, peaks_paired):
        result = np.zeros(len(peaks_paired), dtype=self.dtype)
        for q in self.dtype.names:
            result[q] = peaks_paired[q]
        return result
