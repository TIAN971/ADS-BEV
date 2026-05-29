from typing import Dict, Tuple

import torch
import torch.nn as nn

from .bev_mask_generator import BEVAttentionMaskGenerator, BEVMaskConfig
from .feature_fusion import LanguageGuidedFeatureFusion
from .language_parser import RuleBasedLanguageParser


class LanguageGuidedBEVModule(nn.Module):
    """Language-guided BEV attention enhancement module.

    This module receives a natural language instruction and BEV features, then
    generates an attention mask and enhances the BEV features.
    """

    def __init__(
        self,
        bev_h: int = 200,
        bev_w: int = 200,
        alpha: float = 0.3,
    ) -> None:
        super().__init__()
        self.parser = RuleBasedLanguageParser()
        self.mask_generator = BEVAttentionMaskGenerator(
            BEVMaskConfig(bev_h=bev_h, bev_w=bev_w)
        )
        self.fusion = LanguageGuidedFeatureFusion(alpha=alpha)

    def forward(
        self,
        bev_feature: torch.Tensor,
        instruction: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
        """Run language-guided BEV feature enhancement.

        Args:
            bev_feature: BEV feature tensor with shape [B, C, H, W].
            instruction: Natural language driving instruction.

        Returns:
            enhanced_feature: Enhanced BEV feature.
            attention_mask: Language-guided BEV attention mask.
            metadata: Parsed instruction metadata.
        """
        parsed = self.parser.parse(instruction)
        device = bev_feature.device

        attention_mask = self.mask_generator.generate(parsed, device=device)
        attention_mask = attention_mask.repeat(bev_feature.shape[0], 1, 1, 1)

        if attention_mask.shape[-2:] != bev_feature.shape[-2:]:
            attention_mask = torch.nn.functional.interpolate(
                attention_mask,
                size=bev_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        enhanced_feature = self.fusion(bev_feature, attention_mask)

        metadata = {
            "raw_text": parsed.raw_text,
            "target_classes": parsed.target_classes,
            "spatial_hints": parsed.spatial_hints,
            "priority": parsed.priority,
        }

        return enhanced_feature, attention_mask, metadata
