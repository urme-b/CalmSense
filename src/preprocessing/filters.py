from typing import Tuple, Union

import numpy as np
from scipy import signal

from ..config import FILTER_PARAMS, FS
from ..logging_config import LoggerMixin


class SignalProcessor(LoggerMixin):
    """Butterworth filtering for the slow chest modalities (temperature, respiration).

    ECG and EDA have dedicated processors (``ECGProcessor``, ``EDAProcessor``);
    this class only owns the filters the windowing pipeline still calls.
    """

    def __init__(self, fs: float = FS.CHEST):
        self.fs = fs
        self.logger.debug(f"SignalProcessor initialized with fs={fs} Hz")

    def butterworth_filter(
        self,
        data: np.ndarray,
        cutoff: Union[float, Tuple[float, float]],
        order: int = 4,
        btype: str = "low",
    ) -> np.ndarray:
        nyq = 0.5 * self.fs

        normalized_cutoff: Union[float, Tuple[float, float]]
        if isinstance(cutoff, tuple):
            normalized_cutoff = (cutoff[0] / nyq, cutoff[1] / nyq)
            if normalized_cutoff[0] >= 1.0 or normalized_cutoff[1] >= 1.0:
                raise ValueError(f"Cutoff frequencies {cutoff} exceed Nyquist frequency {nyq} Hz")
        else:
            normalized_cutoff = cutoff / nyq
            if normalized_cutoff >= 1.0:
                raise ValueError(f"Cutoff frequency {cutoff} exceeds Nyquist frequency {nyq} Hz")

        if len(data) == 0:
            return data

        # Very low normalized cutoffs make high-order IIR filters unstable;
        # drop the order and use second-order sections for numerical stability.
        low_norm = (
            min(normalized_cutoff) if isinstance(normalized_cutoff, tuple) else normalized_cutoff
        )
        if low_norm < 0.01:
            order = min(order, 2)

        sos = signal.butter(order, normalized_cutoff, btype=btype, output="sos")
        return signal.sosfiltfilt(sos, data)

    def process_respiration(self, resp: np.ndarray) -> np.ndarray:
        resp = np.asarray(resp).flatten()

        return self.butterworth_filter(
            resp,
            cutoff=(FILTER_PARAMS.RESP_BANDPASS_LOW, FILTER_PARAMS.RESP_BANDPASS_HIGH),
            order=FILTER_PARAMS.RESP_FILTER_ORDER,
            btype="band",
        )

    def process_temperature(self, temp: np.ndarray) -> np.ndarray:
        temp = np.asarray(temp).flatten()

        return self.butterworth_filter(
            temp,
            cutoff=FILTER_PARAMS.TEMP_LOWPASS,
            order=FILTER_PARAMS.TEMP_FILTER_ORDER,
            btype="low",
        )
