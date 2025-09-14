import os
import torch
import torch.nn as nn
import numpy as np
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from utils import calculate_metrics
from utils import solve_heat_transfer
from utils import save_checkpoint as save_checkpoint_to_file

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,
                 epochs: int,
                 model: nn.Module,
                 device: torch.device,
                 train_augmentations: Optional[Any] = None):

        self.epochs = epochs
        self.model = model.to(device)
        self.device = device
        self.train_augmentations = train_augmentations

        self.scaler = torch.cuda.amp.GradScaler()

        self.loss_function = nn.MSELoss()

        self.model_path: Optional[str] = None

    def _apply_physics_constraints(
        self,
        outputs: torch.Tensor,
        laser_power: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prediction = outputs[:, [0]]
        heat_transfer = solve_heat_transfer(
            alpha=outputs[:, [1]],
            thermal_conductivity=outputs[:, [2]],
            radius=outputs[:, [3]],
            efficiency=outputs[:, [4]],
            power=laser_power
        )
        physics_result = heat_transfer + prediction
        return physics_result, prediction

    def _validate_tensor_shapes(
        self,
        prediction: torch.Tensor,
        labels: torch.Tensor
    ) -> None:
        assert prediction.shape == labels.shape, (
            f"Shape mismatch: prediction {prediction.shape} vs labels {labels.shape}"
        )

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        max_epochs: int
    ) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0.0
        dataset_size = len(dataloader.dataset)
        all_labels: List[torch.Tensor] = []
        all_outputs: List[torch.Tensor] = []

        progress_bar = tqdm(
            dataloader,
            desc=f"Train Epoch {epoch+1}/{max_epochs}",
            unit="batch"
        )

        for rgb_images, labels, laser_power in progress_bar:
            rgb_images = rgb_images.to(self.device, non_blocking=True).float()
            labels = labels.to(self.device, non_blocking=True).float()
            laser_power = laser_power.to(self.device, non_blocking=True).float()

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                raw_outputs = self.model(rgb_images)
                physics_result, prediction = self._apply_physics_constraints(
                    raw_outputs, laser_power
                )

                self._validate_tensor_shapes(physics_result, labels)

                batch_loss = self.loss_function(physics_result, labels)

            self.scaler.scale(batch_loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(optimizer)
            self.scaler.update()

            if scheduler is not None:
                scheduler.step()

            batch_size = rgb_images.size(0)
            total_loss += batch_loss.item() * batch_size
            all_labels.append(labels.detach().cpu())
            all_outputs.append(physics_result.detach().cpu())

            current_lr = optimizer.param_groups[0]["lr"]
            progress_bar.set_postfix(
                loss=batch_loss.item(),
                lr=f"{current_lr:.6f}"
            )

        avg_loss = total_loss / dataset_size
        concatenated_labels = torch.cat(all_labels).numpy()
        concatenated_outputs = torch.cat(all_outputs).numpy()
        metrics = calculate_metrics(concatenated_labels, concatenated_outputs)

        return avg_loss, metrics

    def evaluate_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        task_name: str = "validation"
    ) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        dataset_size = len(dataloader.dataset)
        all_labels: List[np.ndarray] = []
        all_outputs: List[np.ndarray] = []

        progress_bar = tqdm(dataloader, desc=task_name.title(), unit="batch")

        with torch.no_grad():
            for rgb_images, labels, laser_power in progress_bar:
                rgb_images = rgb_images.to(self.device, non_blocking=True).float()
                labels = labels.to(self.device, non_blocking=True).float()
                laser_power = laser_power.to(self.device, non_blocking=True).float()

                with torch.cuda.amp.autocast():
                    raw_outputs = self.model(rgb_images)
                    physics_result, prediction = self._apply_physics_constraints(
                        raw_outputs, laser_power
                    )

                    self._validate_tensor_shapes(prediction, labels)

                    batch_loss = self.loss_function(physics_result, labels)

                batch_size = rgb_images.size(0)
                total_loss += batch_loss.item() * batch_size
                all_labels.append(labels.cpu().numpy())
                all_outputs.append(physics_result.cpu().numpy())

                progress_bar.set_postfix(loss=batch_loss.item())

        avg_loss = total_loss / dataset_size
        concatenated_labels = np.concatenate(all_labels, axis=0)
        concatenated_outputs = np.concatenate(all_outputs, axis=0)
        metrics = calculate_metrics(concatenated_labels, concatenated_outputs)

        return avg_loss, metrics

    def save_checkpoint(
        self,
        save_directory: str,
        fold: int
    ) -> None:
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        self.model_path = os.path.join(
            save_directory,
            f"model_fold{fold+1}.pt"
        )
        save_checkpoint_to_file(self.model_path, self.model)
        logger.info(f"Model saved: {self.model_path}")

    def get_model_path(self) -> Optional[str]:
        return self.model_path

    def load_model(self) -> None:
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Model loaded from: {self.model_path}")
        else:
            logger.warning("No model checkpoint found to load.")
