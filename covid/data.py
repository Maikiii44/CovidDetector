from pytorch_lightning import LightningDataModule
from torchvision import transforms, datasets
import torch
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

IMAGE_SIZE = (256, 256)


class CovidDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = os.cpu_count(),
        persistent_workers: bool = True,
    ):
        super().__init__()

        self.train_path = data_path + "/train"
        self.test_path = data_path + "/test"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        # Initialize class_to_idx and classes
        self._class_to_idx = None
        self._classes = None

    def setup(self, stage: str):

        if stage == "fit" and self.train_ds is None:

            fit_transform = transforms.Compose(
                [
                    transforms.Resize(size=IMAGE_SIZE),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            train_dataset = datasets.ImageFolder(
                root=self.train_path, transform=fit_transform
            )

            # Store class_to_idx and classes
            self._class_to_idx = train_dataset.class_to_idx
            self._classes = train_dataset.classes

            # Get the targets
            targets = train_dataset.targets

            # Perform stratified split
            train_indices, val_indices = train_test_split(
                list(range(len(train_dataset))),
                test_size=0.2,
                stratify=targets,
                random_state=42,
            )

            # Create subsets
            self.train_ds = torch.utils.data.Subset(train_dataset, train_indices)
            self.val_ds = torch.utils.data.Subset(train_dataset, val_indices)

        if stage == "test" and self.test_ds is None:

            test_transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

            self.test_ds = datasets.ImageFolder(
                self.test_path, transform=test_transform
            )

            # If test dataset might have different classes
            if self._class_to_idx is None:
                self._class_to_idx = self.test_ds.class_to_idx
                self._classes = self.test_ds.classes

    @property
    def class_to_idx(self):
        return self._class_to_idx

    @property
    def classes(self):
        return self._classes

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )
