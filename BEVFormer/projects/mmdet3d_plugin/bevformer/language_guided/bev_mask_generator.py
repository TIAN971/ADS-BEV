from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from .language_parser import ParsedInstruction


@dataclass
class BEVMaskConfig:
    """Configuration for BEV attention mask generation."""

    bev_h: int = 200
    bev_w: int = 200
    x_range: Tuple[float, float] = (-50.0, 50.0)
    y_range: Tuple[float, float] = (-50.0, 50.0)
    smoothing_kernel_size: int = 9
    min_value: float = 0.0
    max_value: float = 1.0


class BEVAttentionMaskGenerator:
    """Generate BEV spatial attention masks from parsed instructions."""

    def __init__(self, config: BEVMaskConfig) -> None:
        self.config = config

    def generate(self, parsed: ParsedInstruction, device: torch.device) -> torch.Tensor:
        """Generate a BEV attention mask.

        Args:
            parsed: Parsed instruction result.
            device: Target torch device.

        Returns:
            Tensor with shape [1, 1, bev_h, bev_w].
        """
        h, w = self.config.bev_h, self.config.bev_w

        y = torch.linspace(self.config.y_range[0], self.config.y_range[1], h, device=device)
        x = torch.linspace(self.config.x_range[0], self.config.x_range[1], w, device=device)
        grid_y, grid_x = torch.meshgrid(y, x)

        mask = torch.zeros((h, w), device=device)

        if "front" in parsed.spatial_hints:
            mask = torch.maximum(mask, torch.sigmoid((grid_x - 0.0) / 8.0))

        if "rear" in parsed.spatial_hints:
            mask = torch.maximum(mask, torch.sigmoid((-grid_x - 0.0) / 8.0))

        if "left" in parsed.spatial_hints:
            mask = torch.maximum(mask, torch.sigmoid((grid_y - 0.0) / 8.0))

        if "right" in parsed.spatial_hints:
            mask = torch.maximum(mask, torch.sigmoid((-grid_y - 0.0) / 8.0))

        if "near" in parsed.spatial_hints:
            distance = torch.sqrt(grid_x ** 2 + grid_y ** 2)
            near_mask = torch.exp(-(distance ** 2) / (2 * 18.0 ** 2))
            mask = torch.maximum(mask, near_mask)

        if "far" in parsed.spatial_hints:
            distance = torch.sqrt(grid_x ** 2 + grid_y ** 2)
            far_mask = torch.sigmoid((distance - 25.0) / 6.0)
            mask = torch.maximum(mask, far_mask)

        mask = mask * parsed.priority
        mask = torch.clamp(mask, self.config.min_value, self.config.max_value)

        mask = mask[None, None, :, :]
        mask = self._smooth(mask)
        return mask

    def _smooth(self, mask: torch.Tensor) -> torch.Tensor:
        """Apply average smoothing to avoid sharp attention changes."""
        kernel_size = self.config.smoothing_kernel_size
        if kernel_size <= 1:
            return mask

        padding = kernel_size // 2
        return F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
