from typing import Callable, List, Optional, Tuple

import numpy as np

from ...logging_config import get_logger

logger = get_logger(__name__)


class SignalAugmenter:
    def __init__(self, random_state: Optional[int] = None):
        self.rng = np.random.RandomState(random_state)

    def set_random_state(self, seed: int) -> None:
        self.rng = np.random.RandomState(seed)

    def jittering(self, x: np.ndarray, sigma: float = 0.03) -> np.ndarray:

        noise = self.rng.randn(*x.shape) * sigma * np.std(x)
        return x + noise

    def scaling(self, x: np.ndarray, sigma: float = 0.1) -> np.ndarray:

        scale_factor = self.rng.randn() * sigma + 1.0
        return x * scale_factor

    def magnitude_warping(
        self, x: np.ndarray, sigma: float = 0.2, num_knots: int = 4
    ) -> np.ndarray:

        from scipy.interpolate import CubicSpline

        x = np.atleast_2d(x)
        if x.shape[0] == 1:
            x = x.T

        seq_len = x.shape[0]

        knot_xs = np.linspace(0, seq_len - 1, num_knots + 2)
        knot_ys = self.rng.randn(num_knots + 2) * sigma + 1.0

        spline = CubicSpline(knot_xs, knot_ys)
        warp_curve = spline(np.arange(seq_len))

        return x * warp_curve[:, np.newaxis]

    def time_warping(
        self, x: np.ndarray, sigma: float = 0.2, num_knots: int = 4
    ) -> np.ndarray:

        from scipy.interpolate import CubicSpline, interp1d

        x = np.atleast_2d(x)
        if x.shape[0] == 1:
            x = x.T

        seq_len = x.shape[0]
        n_features = x.shape[1] if x.ndim > 1 else 1

        knot_xs = np.linspace(0, seq_len - 1, num_knots + 2)
        knot_ys = knot_xs + self.rng.randn(num_knots + 2) * sigma * seq_len / num_knots

        # Enforce monotonicity
        knot_ys = np.sort(knot_ys)
        knot_ys[0] = 0
        knot_ys[-1] = seq_len - 1

        warp_fn = CubicSpline(knot_xs, knot_ys)
        warped_indices = warp_fn(np.arange(seq_len))
        warped_indices = np.clip(warped_indices, 0, seq_len - 1)

        result = np.zeros_like(x)
        for i in range(n_features):
            interp_fn = interp1d(
                np.arange(seq_len),
                x[:, i] if x.ndim > 1 else x,
                kind="linear",
                fill_value="extrapolate",
            )
            if x.ndim > 1:
                result[:, i] = interp_fn(warped_indices)
            else:
                result = interp_fn(warped_indices)

        return result.squeeze()

    def random_crop(self, x: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:

        from scipy.ndimage import zoom

        seq_len = x.shape[0]
        crop_len = int(seq_len * crop_ratio)

        start = self.rng.randint(0, seq_len - crop_len + 1)
        cropped = x[start : start + crop_len]

        if x.ndim == 1:
            zoom_factor = seq_len / crop_len
            return zoom(cropped, zoom_factor, order=1)
        else:
            zoom_factors = (seq_len / crop_len, 1)
            return zoom(cropped, zoom_factors, order=1)

    def window_slice(self, x: np.ndarray, reduce_ratio: float = 0.9) -> np.ndarray:

        seq_len = x.shape[0]
        target_len = int(seq_len * reduce_ratio)

        start = self.rng.randint(0, seq_len - target_len + 1)
        return x[start : start + target_len]

    def permutation(self, x: np.ndarray, num_segments: int = 4) -> np.ndarray:

        seq_len = x.shape[0]
        segment_len = seq_len // num_segments

        segments = []
        for i in range(num_segments):
            start = i * segment_len
            end = start + segment_len if i < num_segments - 1 else seq_len
            segments.append(x[start:end])

        perm = self.rng.permutation(num_segments)
        permuted = [segments[i] for i in perm]

        return np.concatenate(permuted, axis=0)

    def rotation(self, x: np.ndarray) -> np.ndarray:

        if x.ndim != 2 or x.shape[1] != 3:
            logger.warning("Rotation requires 3D input (seq_len, 3). Skipping.")
            return x

        axis = self.rng.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = self.rng.uniform(0, 2 * np.pi)

        # Rodrigues' formula
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

        return x @ R.T

    def channel_shuffle(self, x: np.ndarray) -> np.ndarray:

        if x.ndim != 2:
            return x

        perm = self.rng.permutation(x.shape[1])
        return x[:, perm]

    def random_masking(self, x: np.ndarray, mask_ratio: float = 0.1) -> np.ndarray:

        mask = self.rng.random(x.shape[0]) > mask_ratio
        if x.ndim == 1:
            return x * mask
        return x * mask[:, np.newaxis]

    def mixup(
        self, x1: np.ndarray, x2: np.ndarray, y1: int, y2: int, alpha: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:

        lam = self.rng.beta(alpha, alpha) if alpha > 0 else 0.5

        mixed_x = lam * x1 + (1 - lam) * x2

        if isinstance(y1, (int, np.integer)):
            num_classes = max(y1, y2) + 1
            y1_onehot = np.zeros(num_classes)
            y2_onehot = np.zeros(num_classes)
            y1_onehot[y1] = 1
            y2_onehot[y2] = 1
            mixed_y = lam * y1_onehot + (1 - lam) * y2_onehot
        else:
            mixed_y = lam * y1 + (1 - lam) * y2

        return mixed_x, mixed_y

    def cutmix(
        self, x1: np.ndarray, x2: np.ndarray, y1: int, y2: int, alpha: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:

        seq_len = x1.shape[0]

        lam = self.rng.beta(alpha, alpha) if alpha > 0 else 0.5
        cut_len = int(seq_len * (1 - lam))

        cut_start = self.rng.randint(0, seq_len - cut_len + 1)
        cut_end = cut_start + cut_len

        mixed_x = x1.copy()
        mixed_x[cut_start:cut_end] = x2[cut_start:cut_end]

        actual_lam = 1 - cut_len / seq_len

        if isinstance(y1, (int, np.integer)):
            num_classes = max(y1, y2) + 1
            y1_onehot = np.zeros(num_classes)
            y2_onehot = np.zeros(num_classes)
            y1_onehot[y1] = 1
            y2_onehot[y2] = 1
            mixed_y = actual_lam * y1_onehot + (1 - actual_lam) * y2_onehot
        else:
            mixed_y = actual_lam * y1 + (1 - actual_lam) * y2

        return mixed_x, mixed_y

    def compose(
        self, augmentations: List[Tuple[str, dict]], p: float = 1.0
    ) -> Callable:

        def transform(x: np.ndarray) -> np.ndarray:
            if self.rng.random() > p:
                return x

            result = x.copy()
            for method_name, kwargs in augmentations:
                method = getattr(self, method_name)
                result = method(result, **kwargs)

            return result

        return transform

    def random_augment(
        self, x: np.ndarray, n_augments: int = 2, magnitude: float = 0.5
    ) -> np.ndarray:

        augment_methods = [
            ("jittering", {"sigma": 0.03 * magnitude}),
            ("scaling", {"sigma": 0.1 * magnitude}),
            ("magnitude_warping", {"sigma": 0.2 * magnitude}),
            ("time_warping", {"sigma": 0.2 * magnitude}),
            ("random_crop", {"crop_ratio": 0.9 + 0.1 * (1 - magnitude)}),
            ("permutation", {"num_segments": max(2, int(4 * magnitude))}),
            ("random_masking", {"mask_ratio": 0.1 * magnitude}),
        ]

        selected = self.rng.choice(len(augment_methods), n_augments, replace=False)

        result = x.copy()
        for idx in selected:
            method_name, kwargs = augment_methods[idx]
            method = getattr(self, method_name)
            result = method(result, **kwargs)

        return result


class ImageAugmenter:
    def __init__(self, random_state: Optional[int] = None):
        self.rng = np.random.RandomState(random_state)

    def random_flip_horizontal(self, img: np.ndarray, p: float = 0.5) -> np.ndarray:
        if self.rng.random() < p:
            return np.flip(img, axis=-1).copy()
        return img

    def random_flip_vertical(self, img: np.ndarray, p: float = 0.5) -> np.ndarray:
        if self.rng.random() < p:
            return np.flip(img, axis=-2).copy()
        return img

    def random_rotation_90(self, img: np.ndarray, p: float = 0.5) -> np.ndarray:
        if self.rng.random() < p:
            k = self.rng.randint(1, 4)
            if img.ndim == 3:
                return np.rot90(img, k, axes=(1, 2)).copy()
            return np.rot90(img, k).copy()
        return img

    def random_crop_resize(
        self, img: np.ndarray, min_crop: float = 0.8, p: float = 0.5
    ) -> np.ndarray:
        from scipy.ndimage import zoom

        if self.rng.random() > p:
            return img

        h, w = img.shape[-2:]
        crop_ratio = self.rng.uniform(min_crop, 1.0)

        new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
        top = self.rng.randint(0, h - new_h + 1)
        left = self.rng.randint(0, w - new_w + 1)

        if img.ndim == 3:
            cropped = img[:, top : top + new_h, left : left + new_w]
            zoom_factors = (1, h / new_h, w / new_w)
        else:
            cropped = img[top : top + new_h, left : left + new_w]
            zoom_factors = (h / new_h, w / new_w)

        return zoom(cropped, zoom_factors, order=1)

    def gaussian_noise(self, img: np.ndarray, sigma: float = 0.02) -> np.ndarray:
        noise = self.rng.randn(*img.shape) * sigma
        return img + noise

    def brightness_contrast(
        self,
        img: np.ndarray,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
    ) -> np.ndarray:
        brightness = self.rng.uniform(*brightness_range)
        contrast = self.rng.uniform(*contrast_range)

        mean = img.mean()
        result = (img - mean) * contrast + mean + (brightness - 1.0)

        return np.clip(result, 0, 1)

    def cutout(
        self, img: np.ndarray, n_holes: int = 1, hole_size: float = 0.1
    ) -> np.ndarray:

        h, w = img.shape[-2:]
        hole_h, hole_w = int(h * hole_size), int(w * hole_size)

        result = img.copy()
        for _ in range(n_holes):
            y = self.rng.randint(0, h - hole_h + 1)
            x = self.rng.randint(0, w - hole_w + 1)

            if result.ndim == 3:
                result[:, y : y + hole_h, x : x + hole_w] = 0
            else:
                result[y : y + hole_h, x : x + hole_w] = 0

        return result

    def compose(
        self, transforms: List[Tuple[Callable, dict]], p: float = 1.0
    ) -> Callable:
        def apply(img: np.ndarray) -> np.ndarray:
            if self.rng.random() > p:
                return img

            result = img.copy()
            for transform, kwargs in transforms:
                result = transform(result, **kwargs)

            return result

        return apply
