import numpy as np
import strax
from strax import DownChunkingPlugin
import straxen
from straxen import units, EventBasics, EventPositions

from ...utils import copy_dtype
from ...samplers import SAMPLERS
from ...plugin import ExhaustPlugin


class EventsSalting(ExhaustPlugin, DownChunkingPlugin, EventPositions, EventBasics):
    __version__ = "0.0.0"
    child_plugin = True
    depends_on = "run_meta"
    provides = "events_salting"
    data_kind = "events_salting"
    save_when = strax.SaveWhen.EXPLICIT

    salting_seed = straxen.URLConfig(
        default=None,
        type=(int, None),
        help="Seed for salting",
    )

    salting_rate = straxen.URLConfig(
        default=20,
        type=(int, float),
        help="Rate of salting in Hz",
    )

    s1_area_range = straxen.URLConfig(
        default=(1, 150),
        type=(list, tuple),
        help="Range of S1 area in salting",
    )

    s2_area_range = straxen.URLConfig(
        default=(1e2, 2e4),
        type=(list, tuple),
        help="Range of S2 area in salting",
    )

    s1_distribution = straxen.URLConfig(
        default="exponential",
        type=str,
        help="S1 distribution shape in salting",
    )

    s2_distribution = straxen.URLConfig(
        default="exponential",
        type=str,
        help="S2 distribution shape in salting",
    )

    veto_length_run_start = straxen.URLConfig(
        default=10**9,
        type=int,
        help="Min time delay in [ns] for events towards the run start boundary",
    )

    veto_length_run_end = straxen.URLConfig(
        default=10**8,
        type=int,
        help="Min time delay in [ns] for events towards the run end boundary",
    )

    s1_n_hits_tight_coincidence = straxen.URLConfig(
        default=2,
        type=int,
        help="Will assign the events's ``s1_n_hits`` and ``s1_tight_coincidence`` by this",
    )

    n_drift_time_window = straxen.URLConfig(
        default=5,
        type=int,
        help="How many max drift time will the event builder extend",
    )

    def _set_posrec_save(self):
        self.posrec_save = []
        self.pos_rec_labels = []

    def refer_dtype(self):
        return strax.merged_dtype(
            [
                strax.to_numpy_dtype(super(EventPositions, self).infer_dtype()),
                strax.to_numpy_dtype(super(EventsSalting, self).infer_dtype()),
            ]
        )

    def infer_dtype(self):
        dtype_reference = self.refer_dtype()
        required_names = ["time", "endtime", "s1_center_time", "s2_center_time"]
        required_names += ["s1_area", "s2_area", "s1_n_hits", "s1_tight_coincidence"]
        required_names += ["x", "y", "z", "drift_time", "s2_x", "s2_y", "z_naive"]
        dtype = copy_dtype(dtype_reference, required_names)
        # since event_number is int64 in event_basics
        dtype += [(("Salting number of events", "salt_number"), np.int64)]
        return dtype

    def setup(self):
        super(EventPositions, self).setup()
        super(EventsSalting, self).setup()

        self.init_rng()

    def init_rng(self):
        """Initialize the random number generator."""
        if self.salting_seed is None:
            self.rng = np.random.default_rng(seed=int(self.run_id))
        else:
            self.rng = np.random.default_rng(seed=self.salting_seed)

    def sample_time(self, start, end):
        """Sample the time according to the start and end of the run."""
        self.event_time_interval = int(units.s // self.salting_rate)

        if units.s / self.salting_rate < self.drift_time_max * self.n_drift_time_window * 2:
            raise ValueError("Salting rate is too high according the drift time window!")

        time = np.arange(
            start + self.veto_length_run_start,
            end - self.veto_length_run_end,
            self.event_time_interval,
        ).astype(np.int64)
        return time

    def inverse_field_distortion(self, x, y, z):
        # TODO: implement detailed inverse field distortion
        return x, y, z

    def set_chunk_splitting(self):
        """Split the rujn into chunks to prevent oversize chunk."""
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

    def sampling(self, start, end):
        """Sample the features of events, (t, x, y, z, S1, S2) et al."""
        time = self.sample_time(start, end)
        self.n_events = len(time)
        self.events_salting = np.empty(self.n_events, dtype=self.dtype)
        self.events_salting["salt_number"] = np.arange(self.n_events)
        self.events_salting["time"] = time
        self.events_salting["endtime"] = time

        self.events_salting["s1_n_hits"] = self.s1_n_hits_tight_coincidence
        self.events_salting["s1_tight_coincidence"] = self.s1_n_hits_tight_coincidence

        theta = self.rng.random(size=self.n_events) * 2 * np.pi - np.pi
        r = np.sqrt(self.rng.random(size=self.n_events)) * straxen.tpc_r
        self.events_salting["x"] = np.cos(theta) * r
        self.events_salting["y"] = np.sin(theta) * r
        self.events_salting["z"] = -self.rng.random(size=self.n_events) * straxen.tpc_z
        s2_x, s2_y, z_naive = self.inverse_field_distortion(
            self.events_salting["x"],
            self.events_salting["y"],
            self.events_salting["z"],
        )
        self.events_salting["s2_x"] = s2_x
        self.events_salting["s2_y"] = s2_y
        self.events_salting["z_naive"] = z_naive
        self.events_salting["drift_time"] = (
            self.electron_drift_velocity * self.electron_drift_time_gate
            - self.events_salting["z_naive"]
        ) / self.electron_drift_velocity

        self.events_salting["s1_center_time"] = time - self.events_salting["drift_time"]
        self.events_salting["s2_center_time"] = time

        s1_area_range = (float(self.s1_area_range[0]), float(self.s1_area_range[1]))
        s2_area_range = (float(self.s2_area_range[0]), float(self.s2_area_range[1]))
        self.events_salting["s1_area"] = SAMPLERS[self.s1_distribution](s1_area_range).sample(
            self.n_events, self.rng
        )
        self.events_salting["s2_area"] = SAMPLERS[self.s2_distribution](s2_area_range).sample(
            self.n_events, self.rng
        )
        # to prevent numerical errors
        self.events_salting["s1_area"] = np.clip(self.events_salting["s1_area"], *s1_area_range)
        self.events_salting["s2_area"] = np.clip(self.events_salting["s2_area"], *s2_area_range)

        if np.any(np.diff(self.events_salting["time"]) <= 0):
            raise ValueError("The time is not strictly increasing!")

        self.set_chunk_splitting()

    def compute(self, run_meta, start, end):
        """Copy and assign the salting events into chunk."""
        self.sampling(start, end)
        for chunk_i in range(len(self.slices)):
            indices = self.slices[chunk_i]

            if chunk_i == 0:
                _start = start
            else:
                _start = self.events_salting["time"][indices[0]] - self.time_left

            if chunk_i == len(self.slices) - 1:
                _end = end
            else:
                _end = self.events_salting["time"][indices[1] - 1] + self.time_right

            yield self.chunk(
                start=_start, end=_end, data=self.events_salting[indices[0] : indices[1]]
            )
