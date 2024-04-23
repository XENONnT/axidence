import warnings
from typing import Tuple
import numpy as np
import utilix
import strax
import straxen
from straxen import units, EventBasics, EventPositions
from straxen.misc import kind_colors


kind_colors.update(
    {
        "salting_events": "#0080ff",
        "salting_peaks": "#00ffff",
    }
)


class SaltingEvents(EventBasics, EventPositions):
    __version__ = "0.0.0"
    depends_on: Tuple = tuple()
    provides = "salting_events"
    data_kind = "salting_events"
    save_when = strax.SaveWhen.EXPLICIT

    salting_seed = straxen.URLConfig(
        default=None,
        type=(int, None),
        help="Seed for salting",
    )

    real_run_start = straxen.URLConfig(
        default=None,
        type=(int, None),
        help="Real start time of run [ns]",
    )

    real_run_end = straxen.URLConfig(
        default=None,
        type=(int, None),
        help="Real start time of run [ns]",
    )

    strict_real_run_time_check = straxen.URLConfig(
        default=True,
        type=bool,
        help="Whether to strictly check the real run time is provided",
    )

    salting_rate = straxen.URLConfig(
        default=10,
        type=(int, float),
        help="Rate of salting in Hz",
    )

    s1_area_range = straxen.URLConfig(
        default=(1, 150),
        type=(list, tuple),
        help="Range of S1 area",
    )

    s2_area_range = straxen.URLConfig(
        default=(1e2, 2e4),
        type=(list, tuple),
        help="Range of S2 area",
    )

    s1_distribution = straxen.URLConfig(
        default="",
        type=str,
        help="S1 distribution shape",
    )

    s2_distribution = straxen.URLConfig(
        default="",
        type=str,
        help="S2 distribution shape",
    )

    veto_length_run_start = straxen.URLConfig(
        default=10**9,
        type=int,
        track=True,
        help="Min time delay in [ns] for events towards the run start boundary",
    )

    veto_length_run_end = straxen.URLConfig(
        default=10**8,
        type=int,
        track=True,
        help="Min time delay in [ns] for events towards the run end boundary",
    )

    s1_n_hits_tight_coincidence = straxen.URLConfig(
        default=2,
        type=int,
        track=True,
        help="Will assign the events's ``s1_n_hits`` and ``s1_tight_coincidence`` by this",
    )

    def _set_posrec_save(self):
        self.posrec_save = []
        self.pos_rec_labels = []

    def infer_dtype(self):
        dtype = []
        dtype_reference = strax.merged_dtype(
            [
                strax.to_numpy_dtype(super().infer_dtype()),
                strax.to_numpy_dtype(super(EventBasics, self).infer_dtype()),
            ]
        )
        for n in (
            "s1_center_time",
            "s2_center_time",
            "s1_area",
            "s2_area",
            "s1_n_hits",
            "s1_tight_coincidence",
            "x",
            "y",
            "z",
            "drift_time",
            "s2_x",
            "s2_y",
            "z_naive",
        ):
            for x in dtype_reference:
                found = False
                if (x[0][1] == n) and (not found):
                    dtype.append(x)
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not find {n} in dtype_reference!")
        # since event_number is int64 in event_basics
        dtype += [(("Salting number of events", "salt_number"), np.int64)]
        return dtype + strax.time_fields

    def init_rng(self):
        if self.salting_seed is None:
            self.rng = np.random.default_rng(seed=int(self.run_id))
        else:
            self.rng = np.random.default_rng(seed=self.salting_seed)

    def init_run_meta(self):
        if self.real_run_start is None or self.real_run_end is None:
            if self.strict_real_run_time_check:
                raise ValueError("Real run start and end times are not provided!")
            else:
                warnings.warn(
                    "Real run start and end times are not provided. Using utilix to get them."
                )
            if straxen.utilix_is_configured():
                coll = utilix.xent_collection()
            else:
                raise ValueError("Utilix is not configured cannot determine run mode.")
            _doc = coll.find_one(
                {"number": int(self.run_id)}, projection={"start": True, "end": True}
            )
            self.run_start = int(_doc["start"].timestamp() * units.s)
            self.run_end = int(_doc["end"].timestamp() * units.s)
        else:
            self.run_start = self.real_run_start
            self.run_end = self.real_run_end

    def sample_time(self):
        self.event_time_interval = units.s // self.salting_rate

        time = np.arange(
            self.run_start + self.veto_length_run_start,
            self.run_end - self.veto_length_run_end,
            self.event_time_interval,
        ).astype(np.int64)
        self.time_left = self.event_time_interval // 2
        self.time_right = self.event_time_interval - self.time_left
        return time

    def inverse_field_distortion(self, x, y, z):
        # TODO:
        return x, y, z

    def set_chunk_splitting(self):
        self.max_n_events_in_chunk = round(
            self.chunk_target_size_mb * 1e6 / self.dtype.itemsize * 0.9
        )
        slices_idx = np.unique(
            np.append(
                np.arange(self.n_events, dtype=int)[:: self.max_n_events_in_chunk], self.n_events
            )
        )
        self.slices = np.vstack([slices_idx[:-1], slices_idx[1:]]).T.astype(int).tolist()

        self.time_left = self.event_time_interval // 2
        self.time_right = self.event_time_interval - self.time_left

    def setup(self):
        super().setup()

        self.init_rng()
        self.init_run_meta()

        time = self.sample_time()
        self.n_events = len(time)
        self.salting_events = np.empty(self.n_events, dtype=self.dtype)
        self.salting_events["salt_number"] = np.arange(self.n_events)
        self.salting_events["time"] = time
        self.salting_events["endtime"] = time
        self.salting_events["s1_center_time"] = time
        self.salting_events["s2_center_time"] = time

        self.salting_events["s1_n_hits"] = self.s1_n_hits_tight_coincidence
        self.salting_events["s1_tight_coincidence"] = self.s1_n_hits_tight_coincidence

        theta = self.rng.random(size=self.n_events) * 2 * np.pi - np.pi
        r = np.sqrt(self.rng.random(size=self.n_events)) * straxen.tpc_r
        self.salting_events["x"] = np.cos(theta) * r
        self.salting_events["y"] = np.sin(theta) * r
        self.salting_events["z"] = -self.rng.random(size=self.n_events) * straxen.tpc_z
        self.salting_events["s2_x"], self.salting_events["s2_y"], self.salting_events["z_naive"] = (
            self.inverse_field_distortion(
                self.salting_events["x"],
                self.salting_events["y"],
                self.salting_events["z"],
            )
        )
        self.salting_events["drift_time"] = (
            -(
                self.salting_events["z_naive"]
                - self.electron_drift_velocity * self.electron_drift_time_gate
            )
            / self.electron_drift_velocity
            + self.electron_drift_time_gate
        )

        self.salting_events["s1_area"] = np.exp(
            self.rng.uniform(
                np.log(self.s1_area_range[0]), np.log(self.s1_area_range[1]), size=self.n_events
            )
        )
        self.salting_events["s2_area"] = np.exp(
            self.rng.uniform(
                np.log(self.s2_area_range[0]), np.log(self.s2_area_range[1]), size=self.n_events
            )
        )

        self.set_chunk_splitting()

    def compute(self, chunk_i):
        indices = self.slices[chunk_i]

        if chunk_i == 0:
            start = self.run_start
        else:
            start = self.salting_events["time"][indices[0]] - self.time_left

        if chunk_i == len(self.slices) - 1:
            end = self.run_end
        else:
            end = self.salting_events["time"][indices[1] - 1] + self.time_right
        return self.chunk(start=start, end=end, data=self.salting_events[indices[0] : indices[1]])

    def is_ready(self, chunk_i):
        if chunk_i < len(self.slices):
            return True
        else:
            return False

    def source_finished(self):
        return True
