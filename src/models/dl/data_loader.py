from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from ...logging_config import get_logger

logger = get_logger(__name__)


class StressDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
    ):

        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.subject_ids = (
            subject_ids if subject_ids is None else np.asarray(subject_ids)
        )
        self.transform = transform
        self.dtype = dtype

        assert len(self.features) == len(self.labels), (
            "Features and labels must have same length"
        )

        if self.subject_ids is not None:
            assert len(self.subject_ids) == len(self.features), (
                "Subject IDs must have same length as features"
            )

        logger.debug(
            f"Created StressDataset with {len(self)} samples, "
            f"{self.features.shape[1]} features, "
            f"{len(np.unique(self.labels))} classes"
        )

    def __len__(self) -> int:

        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.features[idx]
        y = self.labels[idx]

        if self.transform is not None:
            x = self.transform(x)

        x = torch.tensor(x, dtype=self.dtype)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

    def get_subject(self, idx: int) -> Optional[Any]:

        if self.subject_ids is None:
            return None
        return self.subject_ids[idx]

    def get_class_weights(self) -> torch.Tensor:

        class_counts = np.bincount(self.labels)
        total = len(self.labels)
        weights = total / (len(class_counts) * class_counts)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:

        class_weights = self.get_class_weights()
        sample_weights = class_weights[self.labels]
        return sample_weights


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        max_length: Optional[int] = None,
    ):

        self.sequences = sequences
        self.labels = np.asarray(labels, dtype=np.int64)
        self.subject_ids = subject_ids
        self.transform = transform
        self.max_length = max_length

        # Handle list of variable-length
        if isinstance(sequences, list):
            self._variable_length = True
            self.lengths = [len(seq) for seq in sequences]
            if max_length is None:
                self.max_length = max(self.lengths)
        else:
            self._variable_length = False
            self.sequences = np.asarray(sequences, dtype=np.float32)
            self.max_length = sequences.shape[1]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        seq = self.sequences[idx]
        label = self.labels[idx]

        if self.transform is not None:
            seq = self.transform(seq)

        seq = np.asarray(seq, dtype=np.float32)

        # Pad or truncate to
        if len(seq) < self.max_length:
            pad_length = self.max_length - len(seq)
            seq = np.pad(seq, ((0, pad_length), (0, 0)), mode="constant")
        elif len(seq) > self.max_length:
            seq = seq[: self.max_length]

        x = torch.tensor(seq, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return x, y

    def get_sequence_lengths(self) -> List[int]:

        if self._variable_length:
            return self.lengths
        return [self.max_length] * len(self)


class ImageStressDataset(Dataset):
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
        augmentation: Optional[Callable] = None,
    ):

        self.images = np.asarray(images, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.subject_ids = subject_ids
        self.transform = transform
        self.augmentation = augmentation
        self._training = True

        # Add channel dimension if
        if self.images.ndim == 3:
            self.images = self.images[:, np.newaxis, :, :]

        logger.debug(
            f"Created ImageStressDataset with {len(self)} images, "
            f"shape {self.images.shape[1:]}"
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        img = self.images[idx].copy()
        label = self.labels[idx]

        # Apply augmentation during training
        if self._training and self.augmentation is not None:
            img = self.augmentation(img)

        # Apply preprocessing transform
        if self.transform is not None:
            img = self.transform(img)

        x = torch.tensor(img, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return x, y

    def train(self) -> None:

        self._training = True

    def eval(self) -> None:

        self._training = False


class MultimodalDataset(Dataset):
    def __init__(
        self,
        modality_data: Dict[str, np.ndarray],
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
        transforms: Optional[Dict[str, Callable]] = None,
    ):

        self.modality_data = {
            k: np.asarray(v, dtype=np.float32) for k, v in modality_data.items()
        }
        self.modality_names = list(modality_data.keys())
        self.labels = np.asarray(labels, dtype=np.int64)
        self.subject_ids = subject_ids
        self.transforms = transforms or {}

        # Verify all modalities have
        n_samples = len(self.labels)
        for name, data in self.modality_data.items():
            assert len(data) == n_samples, (
                f"Modality {name} has {len(data)} samples, expected {n_samples}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:

        x = {}
        for name in self.modality_names:
            data = self.modality_data[name][idx]

            if name in self.transforms:
                data = self.transforms[name](data)

            x[name] = torch.tensor(data, dtype=torch.float32)

        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x, y


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray] = None,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.0,
    num_workers: int = 0,
    use_weighted_sampler: bool = True,
    seed: int = 42,
) -> Dict[str, DataLoader]:

    np.random.seed(seed)

    n_samples = len(X)
    indices = np.arange(n_samples)

    # Stratified split
    from sklearn.model_selection import train_test_split

    if test_split > 0:
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_split, stratify=y, random_state=seed
        )
    else:
        train_val_idx = indices
        test_idx = None

    if val_split > 0:
        relative_val = val_split / (1 - test_split) if test_split > 0 else val_split
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val,
            stratify=y[train_val_idx],
            random_state=seed,
        )
    else:
        train_idx = train_val_idx
        val_idx = None

    # Create datasets
    train_dataset = StressDataset(X[train_idx], y[train_idx])

    # Create weighted sampler for
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_dataset), replacement=True
        )
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    }

    if val_idx is not None:
        val_dataset = StressDataset(X[val_idx], y[val_idx])
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    if test_idx is not None:
        test_dataset = StressDataset(X[test_idx], y[test_idx])
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    logger.info(
        f"Created DataLoaders: train={len(train_idx)}, "
        f"val={len(val_idx) if val_idx is not None else 0}, "
        f"test={len(test_idx) if test_idx is not None else 0}"
    )

    return loaders


def create_loso_loaders(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_subject: Union[int, str],
    batch_size: int = 32,
    num_workers: int = 0,
    use_weighted_sampler: bool = True,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:

    groups = np.asarray(groups)

    # Split indices
    test_mask = groups == test_subject
    train_val_mask = ~test_mask

    test_idx = np.where(test_mask)[0]
    train_val_idx = np.where(train_val_mask)[0]

    if len(test_idx) == 0:
        raise ValueError(f"No samples found for subject {test_subject}")

    # Further split train/val from
    if val_split > 0:
        rng = np.random.RandomState(42)
        rng.shuffle(train_val_idx)
        val_size = int(len(train_val_idx) * val_split)
        val_idx = train_val_idx[:val_size]
        train_idx = train_val_idx[val_size:]
    else:
        train_idx = train_val_idx
        val_idx = None

    # Create datasets
    train_dataset = StressDataset(X[train_idx], y[train_idx])
    test_dataset = StressDataset(X[test_idx], y[test_idx])

    # Weighted sampler
    if use_weighted_sampler:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_dataset), replacement=True
        )
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_idx is not None and len(val_idx) > 0:
        val_dataset = StressDataset(X[val_idx], y[val_idx])
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    logger.debug(
        f"LOSO split for subject {test_subject}: "
        f"train={len(train_idx)}, val={len(val_idx) if val_idx is not None else 0}, "
        f"test={len(test_idx)}"
    )

    return train_loader, val_loader, test_loader


def create_sequence_loaders(
    sequences: np.ndarray,
    labels: np.ndarray,
    groups: Optional[np.ndarray] = None,
    batch_size: int = 32,
    val_split: float = 0.2,
    max_length: Optional[int] = None,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:

    n_samples = len(labels)
    indices = np.arange(n_samples)

    from sklearn.model_selection import train_test_split

    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=labels, random_state=42
    )

    train_dataset = SequenceDataset(
        sequences[train_idx], labels[train_idx], max_length=max_length
    )
    val_dataset = SequenceDataset(
        sequences[val_idx], labels[val_idx], max_length=max_length
    )

    # Get sample weights
    sample_weights = StressDataset(
        np.zeros((len(train_idx), 1)), labels[train_idx]
    ).get_sample_weights()

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_dataset), replacement=True
    )

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }


def collate_variable_length(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    sequences, labels = zip(*batch)

    # Get lengths
    lengths = torch.tensor([len(seq) for seq in sequences])
    max_len = lengths.max().item()

    # Pad sequences
    padded = torch.zeros(len(sequences), max_len, sequences[0].size(-1))
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq

    labels = torch.stack(labels)

    return padded, labels, lengths
