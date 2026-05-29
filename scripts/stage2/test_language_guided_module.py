import sys
from pathlib import Path

import torch

bevformer_root = Path(__file__).resolve().parents[2] / "BEVFormer"
sys.path.insert(0, str(bevformer_root))

from projects.mmdet3d_plugin.bevformer.language_guided import LanguageGuidedBEVModule


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module = LanguageGuidedBEVModule(bev_h=200, bev_w=200, alpha=0.3).to(device)
    module.eval()

    bev_feature = torch.randn(1, 64, 200, 200, device=device)

    instructions = [
        "focus on pedestrians in front of the ego vehicle",
        "pay attention to cars on the left",
        "watch nearby traffic cones",
        "focus on vehicles behind the ego car",
    ]

    for instruction in instructions:
        with torch.no_grad():
            enhanced, mask, metadata = module(bev_feature, instruction)

        print("=" * 80)
        print("Instruction:", instruction)
        print("Metadata:", metadata)
        print("Input feature shape:", tuple(bev_feature.shape))
        print("Enhanced feature shape:", tuple(enhanced.shape))
        print("Mask shape:", tuple(mask.shape))
        print("Mask min/max:", float(mask.min()), float(mask.max()))

        assert enhanced.shape == bev_feature.shape
        assert mask.shape == (1, 1, 200, 200)

    print("=" * 80)
    print("Language-guided BEV module functional test passed.")


if __name__ == "__main__":
    main()
