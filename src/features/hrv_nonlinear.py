from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats

from ..config import FEATURE_PARAMS
from ..logging_config import LoggerMixin


class HRVNonlinearExtractor(LoggerMixin):
    def __init__(self, min_rr_count: int = 50):
        self.min_rr_count = min_rr_count
        self.logger.debug(f"HRVNonlinearExtractor initialized, min_rr={min_rr_count}")

    def _validate_input(self, rr_intervals: np.ndarray) -> Optional[np.ndarray]:
        rr = np.asarray(rr_intervals).flatten()
        rr = rr[np.isfinite(rr)]

        if len(rr) < self.min_rr_count:
            self.logger.warning(
                f"Insufficient RR intervals: {len(rr)} < {self.min_rr_count}"
            )
            return None

        rr = rr[(rr >= 200) & (rr <= 2500)]

        if len(rr) < self.min_rr_count:
            return None

        return rr

    def compute_sample_entropy(
        self, rr: np.ndarray, m: int = 2, r: float = 0.2
    ) -> float:
        n = len(rr)
        if n < m + 2:
            return np.nan

        r_val = r * np.std(rr)

        def _count_matches(template_len: int) -> int:
            count = 0
            for i in range(n - template_len):
                for j in range(i + 1, n - template_len):
                    dist = np.max(
                        np.abs(rr[i : i + template_len] - rr[j : j + template_len])
                    )
                    if dist < r_val:
                        count += 1
            return count

        b = _count_matches(m)
        a = _count_matches(m + 1)

        if b == 0:
            return np.nan

        return float(
            -np.log((a + FEATURE_PARAMS.EPSILON) / (b + FEATURE_PARAMS.EPSILON))
        )

    def compute_approximate_entropy(
        self, rr: np.ndarray, m: int = 2, r: float = 0.2
    ) -> float:
        n = len(rr)
        if n < m + 2:
            return np.nan

        r_val = r * np.std(rr)

        def _phi(template_len: int) -> float:
            patterns = np.array(
                [rr[i : i + template_len] for i in range(n - template_len + 1)]
            )
            n_patterns = len(patterns)

            if n_patterns == 0:
                return 0.0

            counts = np.zeros(n_patterns)
            for i in range(n_patterns):
                for j in range(n_patterns):
                    dist = np.max(np.abs(patterns[i] - patterns[j]))
                    if dist <= r_val:
                        counts[i] += 1

            probs = counts / n_patterns
            return np.mean(np.log(probs + FEATURE_PARAMS.EPSILON))

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)

        return float(phi_m - phi_m1)

    def compute_dfa(
        self, rr: np.ndarray, scale_min: int = 4, scale_max: int = 64
    ) -> Tuple[float, float]:
        n = len(rr)
        if n < scale_max:
            return np.nan, np.nan

        rr_mean = np.mean(rr)
        y = np.cumsum(rr - rr_mean)

        scales = np.logspace(np.log10(scale_min), np.log10(scale_max), 15).astype(int)
        scales = np.unique(scales)
        scales = scales[scales >= 4]

        fluctuations = []

        for scale in scales:
            n_windows = n // scale

            if n_windows < 2:
                continue

            f_squared = []

            for i in range(n_windows):
                start = i * scale
                end = start + scale
                segment = y[start:end]

                x = np.arange(len(segment))
                coeffs = np.polyfit(x, segment, 1)
                trend = np.polyval(coeffs, x)

                f_squared.append(np.mean((segment - trend) ** 2))

            if len(f_squared) > 0:
                fluctuations.append((scale, np.sqrt(np.mean(f_squared))))

        if len(fluctuations) < 4:
            return np.nan, np.nan

        scales_used = np.array([f[0] for f in fluctuations])
        fluct_values = np.array([f[1] for f in fluctuations])

        log_scales = np.log10(scales_used)
        log_fluct = np.log10(fluct_values + FEATURE_PARAMS.EPSILON)

        # Alpha1: short-term (4-16 beats)
        mask_alpha1 = (scales_used >= 4) & (scales_used <= 16)
        if np.sum(mask_alpha1) >= 2:
            slope1, _, _, _, _ = stats.linregress(
                log_scales[mask_alpha1], log_fluct[mask_alpha1]
            )
            alpha1 = float(slope1)
        else:
            alpha1 = np.nan

        # Alpha2: long-term (16-64 beats)
        mask_alpha2 = (scales_used >= 16) & (scales_used <= 64)
        if np.sum(mask_alpha2) >= 2:
            slope2, _, _, _, _ = stats.linregress(
                log_scales[mask_alpha2], log_fluct[mask_alpha2]
            )
            alpha2 = float(slope2)
        else:
            alpha2 = np.nan

        return alpha1, alpha2

    def compute_poincare(self, rr: np.ndarray) -> Dict[str, float]:
        if len(rr) < 3:
            return {
                "SD1": np.nan,
                "SD2": np.nan,
                "SD1_SD2_ratio": np.nan,
                "CSI": np.nan,
                "CVI": np.nan,
            }

        rr_n = rr[:-1]
        rr_n1 = rr[1:]

        diff = rr_n1 - rr_n

        sd1 = float(np.std(diff, ddof=1) / np.sqrt(2))
        sdnn = np.std(rr, ddof=1)

        sd2_squared = 2 * sdnn**2 - sd1**2
        sd2 = float(np.sqrt(max(sd2_squared, 0)))

        if sd2 > FEATURE_PARAMS.EPSILON:
            sd1_sd2_ratio = float(sd1 / sd2)
        else:
            sd1_sd2_ratio = np.nan

        # CSI = SD2/SD1 (sympathetic
        if sd1 > FEATURE_PARAMS.EPSILON:
            csi = float(sd2 / sd1)
        else:
            csi = np.nan

        # CVI = log10(SD1 *
        if sd1 > 0 and sd2 > 0:
            cvi = float(np.log10(sd1 * sd2))
        else:
            cvi = np.nan

        return {
            "SD1": sd1,
            "SD2": sd2,
            "SD1_SD2_ratio": sd1_sd2_ratio,
            "CSI": csi,
            "CVI": cvi,
        }

    def compute_rqa_determinism(
        self,
        rr: np.ndarray,
        embedding_dim: int = 10,
        time_delay: int = 1,
        radius: float = 0.2,
        min_line_length: int = 2,
    ) -> float:
        n = len(rr)
        if n < embedding_dim + time_delay * embedding_dim:
            return np.nan

        dist_matrix = np.abs(rr[:, np.newaxis] - rr[np.newaxis, :])
        threshold = radius * np.max(dist_matrix)
        recurrence_matrix = (dist_matrix < threshold).astype(int)

        total_recurrence = np.sum(recurrence_matrix) - n

        if total_recurrence == 0:
            return 0.0

        diagonal_points = 0

        for k in range(1, n):
            diag = np.diag(recurrence_matrix, k)
            runs = np.diff(np.concatenate([[0], diag, [0]]))
            run_starts = np.where(runs == 1)[0]
            run_ends = np.where(runs == -1)[0]
            run_lengths = run_ends - run_starts

            for length in run_lengths:
                if length >= min_line_length:
                    diagonal_points += length

        diagonal_points *= 2  # both sides of diagonal

        det = diagonal_points / total_recurrence
        return float(min(det, 1.0))

    def extract_all(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        features = {
            "SampEn": np.nan,
            "ApEn": np.nan,
            "DFA_alpha1": np.nan,
            "DFA_alpha2": np.nan,
            "SD1": np.nan,
            "SD2": np.nan,
            "SD1_SD2_ratio": np.nan,
            "CSI": np.nan,
            "CVI": np.nan,
            "RQA_DET": np.nan,
        }

        rr = self._validate_input(rr_intervals)
        if rr is None:
            self.logger.warning("Invalid input, returning NaN features")
            return features

        try:
            features["SampEn"] = self.compute_sample_entropy(rr, m=2, r=0.2)
            features["ApEn"] = self.compute_approximate_entropy(rr, m=2, r=0.2)

            alpha1, alpha2 = self.compute_dfa(rr)
            features["DFA_alpha1"] = alpha1
            features["DFA_alpha2"] = alpha2

            poincare = self.compute_poincare(rr)
            features.update(poincare)

            rr_sample = rr[: min(200, len(rr))]  # limit for speed
            features["RQA_DET"] = self.compute_rqa_determinism(rr_sample)

            self.logger.debug(
                f"Extracted 10 nonlinear features, "
                f"SampEn={features['SampEn']:.3f}, DFA_alpha1={features['DFA_alpha1']:.3f}"
                if np.isfinite(features["SampEn"])
                else "Extracted 10 nonlinear features"
            )
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")

        return features

    def get_feature_descriptions(self) -> Dict[str, str]:
        return {
            "SampEn": "Sample entropy (complexity, m=2, r=0.2*SD)",
            "ApEn": "Approximate entropy (regularity, m=2, r=0.2*SD)",
            "DFA_alpha1": "DFA short-term scaling exponent (4-16 beats)",
            "DFA_alpha2": "DFA long-term scaling exponent (16-64 beats)",
            "SD1": "Poincaré plot width - short-term variability (ms)",
            "SD2": "Poincaré plot length - long-term variability (ms)",
            "SD1_SD2_ratio": "SD1/SD2 ratio",
            "CSI": "Cardiac Sympathetic Index (SD2/SD1)",
            "CVI": "Cardiac Vagal Index (log10(SD1*SD2))",
            "RQA_DET": "Recurrence quantification determinism (0-1)",
        }
