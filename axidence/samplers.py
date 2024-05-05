import numpy as np
import strax

export, __all__ = strax.exporter()
__all__.extend(["SAMPLERS"])


class Sampler:
    """A class for
    1. sampling data from a distribution
    2. reweight the sampled data given a reference distribution
    """

    def __init__(self, interval, n_bins=None):
        if not isinstance(interval, tuple):
            raise ValueError(f"interval must be tuple, not {type(interval)}, got {interval}!")
        if len(interval) != 2:
            raise ValueError(f"interval must have 2 elements, got {len(interval)}!")
        self.interval = interval
        self.n_bins = n_bins

    def transform(self):
        raise NotImplementedError

    def inverse_transform(self):
        raise NotImplementedError

    def sample(self, n_events, rng):
        return self.inverse_transform(rng.uniform(*self.transform(self.interval), size=n_events))

    @property
    def bins(self):
        if not isinstance(self.n_bins, int):
            raise ValueError(f"n_bins must be int, not {type(self.n_bins)}, got {self.n_bins}!")
        return self.inverse_transform(np.linspace(*self.transform(self.interval), self.n_bins + 1))

    def reweight(self, x, reference, reference_weights=None):
        h_x = np.histogram(x, bins=self.bins)[0]
        h_reference = np.histogram(reference, bins=self.bins, weights=reference_weights)[0]
        _weights = h_reference / h_x
        indices = np.clip(np.digitize(x, self.bins) - 1, 0, len(self.bins) - 2)
        weights = np.where(
            (x > self.bins[0]) & (x < self.bins[-1]),
            _weights[indices],
            0.0,
        )
        return weights


class UniformSampler(Sampler):
    def transform(self, interval):
        return interval

    def inverse_transform(self, x):
        return x


class ExponentialSampler(Sampler):
    def transform(self, interval):
        return np.log(interval)

    def inverse_transform(self, x):
        return np.exp(x)


SAMPLERS = {
    "uniform": UniformSampler,
    "exponential": ExponentialSampler,
}
