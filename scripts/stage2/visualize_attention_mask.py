import sys
from pathlib import Path

import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bevformer_root = Path(__file__).resolve().parents[2] / "BEVFormer"
sys.path.insert(0, str(bevformer_root))

from projects.mmdet3d_plugin.bevformer.language_guided import LanguageGuidedBEVModule


def save_mask(mask: torch.Tensor, title: str, out_file: Path) -> None:
    mask_np = mask.squeeze().detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mask_np, origin="lower", extent=[-50, 50, -50, 50])
    ax.set_title(title)
    ax.set_xlabel("x / forward direction")
    ax.set_ylabel("y / lateral direction")
    ax.scatter([0], [0], marker="x")
    ax.text(0, 0, "ego")
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(__file__).resolve().parents[2] / "docs" / "visualizations" / "stage2_attention_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    module = LanguageGuidedBEVModule(bev_h=200, bev_w=200, alpha=0.3).to(device)
    module.eval()

    bev_feature = torch.randn(1, 64, 200, 200, device=device)

    instructions = [
        "focus on pedestrians in front of the ego vehicle",
        "pay attention to cars on the left",
        "watch nearby traffic cones",
        "focus on vehicles behind the ego car",
        "focus on distant trucks on the right",
    ]

    summary_lines = ["# Stage 2 BEV Attention Mask Visualization\n\n"]

    for idx, instruction in enumerate(instructions):
        with torch.no_grad():
            _, mask, metadata = module(bev_feature, instruction)

        out_file = out_dir / f"attention_mask_{idx:02d}.png"
        save_mask(mask, instruction, out_file)

        summary_lines.append(f"## Sample {idx}\n\n")
        summary_lines.append(f"- Instruction: `{instruction}`\n")
        summary_lines.append(f"- Parsed metadata: `{metadata}`\n")
        summary_lines.append(f"- Output image: `{out_file.name}`\n\n")

        print("Saved:", out_file)

    summary_path = out_dir / "attention_mask_summary.md"
    summary_path.write_text("".join(summary_lines), encoding="utf-8")
    print("Summary saved:", summary_path)


if __name__ == "__main__":
    main()
