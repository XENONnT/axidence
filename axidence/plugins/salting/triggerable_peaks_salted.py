from straxen import TriggerablePeakBasics


class TriggerablePeaksSalted(TriggerablePeakBasics):
    __version__ = "0.0.0"
    depends_on = ("peaks_salted", "peak_shadow_salted", "peak_se_score_salted")
    provides = "triggerable_peaks_salted"
    child_plugin = True

    def infer_dtype(self):
        return self.deps["peaks_salted"].dtype_for("peaks_salted")

    def compute(self, peaks_salted):
        return super().compute(peaks_salted)
