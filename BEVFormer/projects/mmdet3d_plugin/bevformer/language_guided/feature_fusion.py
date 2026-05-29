import torch
import torch.nn as nn


class LanguageGuidedFeatureFusion(nn.Module):
    """Fuse BEV features with a language-guided attention mask."""

    def __init__(self, alpha: float = 0.3) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, bev_feature: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Enhance BEV features using attention mask.

        Args:
            bev_feature: BEV feature tensor with shape [B, C, H, W].
            attention_mask: Attention mask with shape [B, 1, H, W].

        Returns:
            Enhanced BEV feature tensor with shape [B, C, H, W].
        """
        if bev_feature.dim() != 4:
            raise ValueError(f"bev_feature must be 4D, got shape {bev_feature.shape}")

        if attention_mask.dim() != 4:
            raise ValueError(f"attention_mask must be 4D, got shape {attention_mask.shape}")

        if bev_feature.shape[0] != attention_mask.shape[0]:
            raise ValueError("Batch size mismatch between bev_feature and attention_mask")

        if bev_feature.shape[-2:] != attention_mask.shape[-2:]:
            raise ValueError("Spatial size mismatch between bev_feature and attention_mask")

        return bev_feature * (1.0 + self.alpha * attention_mask)
