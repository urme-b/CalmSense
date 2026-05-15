from typing import Optional

import numpy as np

from ..logging_config import LoggerMixin


class SignalImageEncoder(LoggerMixin):
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.logger.debug(f"SignalImageEncoder initialized, size={image_size}")

    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal).flatten()

        if len(signal) == 0:
            return np.array([])

        min_val = np.min(signal)
        max_val = np.max(signal)

        if max_val - min_val < 1e-10:
            return np.zeros_like(signal)

        normalized = 2 * (signal - min_val) / (max_val - min_val) - 1

        return np.clip(normalized, -1, 1)

    def _paa_transform(self, signal: np.ndarray, n_segments: int) -> np.ndarray:

        n = len(signal)
        if n == n_segments:
            return signal
        if n < n_segments:
            indices = np.linspace(0, n - 1, n_segments)
            return np.interp(indices, np.arange(n), signal)

        segment_size = n // n_segments
        paa = np.zeros(n_segments)

        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size if i < n_segments - 1 else n
            paa[i] = np.mean(signal[start:end])

        return paa

    def encode_gasf(
        self, signal: np.ndarray, image_size: Optional[int] = None
    ) -> np.ndarray:

        size = image_size or self.image_size

        signal = np.asarray(signal).flatten()
        if len(signal) < 2:
            self.logger.warning("Signal too short for GASF encoding")
            return np.zeros((size, size))

        normalized = self._normalize_signal(signal)
        paa = self._paa_transform(normalized, size)

        cos_phi = paa
        sin_phi = np.sqrt(np.clip(1 - paa**2, 0, 1))

        gasf = np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)

        self.logger.debug(f"GASF encoded: shape={gasf.shape}")
        return gasf

    def encode_gadf(
        self, signal: np.ndarray, image_size: Optional[int] = None
    ) -> np.ndarray:

        size = image_size or self.image_size

        signal = np.asarray(signal).flatten()
        if len(signal) < 2:
            self.logger.warning("Signal too short for GADF encoding")
            return np.zeros((size, size))

        normalized = self._normalize_signal(signal)
        paa = self._paa_transform(normalized, size)

        cos_phi = paa
        sin_phi = np.sqrt(np.clip(1 - paa**2, 0, 1))

        gadf = np.outer(sin_phi, cos_phi) - np.outer(cos_phi, sin_phi)

        self.logger.debug(f"GADF encoded: shape={gadf.shape}")
        return gadf

    def encode_mtf(
        self, signal: np.ndarray, image_size: Optional[int] = None, n_bins: int = 8
    ) -> np.ndarray:

        size = image_size or self.image_size

        signal = np.asarray(signal).flatten()
        if len(signal) < 2:
            self.logger.warning("Signal too short for MTF encoding")
            return np.zeros((size, size))

        paa = self._paa_transform(signal, size)

        bin_edges = np.percentile(paa, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-10
        binned = np.digitize(paa, bin_edges[:-1]) - 1
        binned = np.clip(binned, 0, n_bins - 1)

        transition_matrix = np.zeros((n_bins, n_bins))
        for i in range(len(binned) - 1):
            transition_matrix[binned[i], binned[i + 1]] += 1

        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums

        mtf = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                mtf[i, j] = transition_matrix[binned[i], binned[j]]

        self.logger.debug(f"MTF encoded: shape={mtf.shape}")
        return mtf

    def encode_rgb(
        self, signal: np.ndarray, image_size: Optional[int] = None
    ) -> np.ndarray:

        size = image_size or self.image_size

        gasf = self.encode_gasf(signal, size)
        gadf = self.encode_gadf(signal, size)
        mtf = self.encode_mtf(signal, size)

        def normalize_image(img):
            min_val = np.min(img)
            max_val = np.max(img)
            if max_val - min_val < 1e-10:
                return np.zeros_like(img)
            return (img - min_val) / (max_val - min_val)

        gasf_norm = normalize_image(gasf)
        gadf_norm = normalize_image(gadf)
        mtf_norm = normalize_image(mtf)

        rgb = np.stack([gasf_norm, gadf_norm, mtf_norm], axis=-1)

        self.logger.debug(f"RGB encoded: shape={rgb.shape}")
        return rgb

    def encode_recurrence_plot(
        self,
        signal: np.ndarray,
        image_size: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        size = image_size or self.image_size

        signal = np.asarray(signal).flatten()
        if len(signal) < 2:
            return np.zeros((size, size))

        paa = self._paa_transform(signal, size)

        dist_matrix = np.abs(paa[:, np.newaxis] - paa[np.newaxis, :])

        if threshold is None:
            threshold = 0.2 * np.std(paa)

        rp = (dist_matrix < threshold).astype(float)

        self.logger.debug(f"Recurrence plot encoded: shape={rp.shape}")
        return rp

    def batch_encode(self, signals: np.ndarray, method: str = "gasf") -> np.ndarray:
        signals = np.asarray(signals)
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)

        n_signals = signals.shape[0]

        encoders = {
            "gasf": self.encode_gasf,
            "gadf": self.encode_gadf,
            "mtf": self.encode_mtf,
            "rgb": self.encode_rgb,
        }
        if method not in encoders:
            raise ValueError(
                f"Unknown encoding method: {method}. Use one of {list(encoders.keys())}"
            )
        encode_func = encoders[method]

        first = encode_func(signals[0])
        output_shape = (n_signals,) + first.shape
        encoded = np.zeros(output_shape)
        encoded[0] = first

        for i in range(1, n_signals):
            encoded[i] = encode_func(signals[i])

        self.logger.info(f"Batch encoded {n_signals} signals using {method}")
        return encoded
