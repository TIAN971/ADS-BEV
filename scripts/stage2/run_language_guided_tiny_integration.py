import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import mmcv
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

bevformer_root = Path(__file__).resolve().parents[2] / "BEVFormer"
sys.path.insert(0, str(bevformer_root))

from projects.mmdet3d_plugin.bevformer.language_guided import LanguageGuidedBEVModule


NUSCENES_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]


def to_numpy(x):
    """Convert tensor-like objects to numpy arrays."""
    if x is None:
        return None
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def get_pred_dict(result_item):
    """Get BEVFormer prediction dict from one result item."""
    if isinstance(result_item, dict) and "pts_bbox" in result_item:
        return result_item["pts_bbox"]
    return result_item


def box_corners_bev(box: np.ndarray) -> np.ndarray:
    """Convert a 3D box into four BEV corners."""
    x, y = float(box[0]), float(box[1])
    dx, dy = float(box[3]), float(box[4])
    yaw = float(box[6]) if len(box) > 6 else 0.0

    local = np.array([
        [dx / 2, dy / 2],
        [dx / 2, -dy / 2],
        [-dx / 2, -dy / 2],
        [-dx / 2, dy / 2],
    ])

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    corners = local @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y
    return corners


def draw_boxes(
    ax,
    boxes: Optional[np.ndarray],
    labels: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    score_thr: float = 0.2,
    top_k: int = 30,
    linestyle: str = "-",
    linewidth: float = 1.2,
) -> int:
    """Draw prediction boxes on BEV figure."""
    if boxes is None or len(boxes) == 0:
        return 0

    boxes = np.asarray(boxes)

    if scores is not None:
        scores = np.asarray(scores)
        keep = np.where(scores >= score_thr)[0]
        if len(keep) > top_k:
            keep = keep[np.argsort(scores[keep])[-top_k:]]
    else:
        keep = np.arange(min(len(boxes), top_k))

    count = 0
    for idx in keep:
        box = boxes[idx]
        corners = box_corners_bev(box)
        poly = Polygon(
            corners[:, :2],
            closed=True,
            fill=False,
            linestyle=linestyle,
            linewidth=linewidth,
        )
        ax.add_patch(poly)

        label_text = "pred"
        if labels is not None and idx < len(labels):
            label_id = int(labels[idx])
            if 0 <= label_id < len(NUSCENES_CLASSES):
                label_text = NUSCENES_CLASSES[label_id]

        if scores is not None and idx < len(scores):
            label_text += f" {float(scores[idx]):.2f}"

        ax.text(float(box[0]), float(box[1]), label_text, fontsize=6)
        count += 1

    return count


def draw_gt_boxes(ax, info: Dict, max_gt: int = 80) -> int:
    """Draw GT boxes from nuScenes info."""
    if "gt_boxes" not in info:
        return 0

    boxes = np.asarray(info["gt_boxes"])
    names = info.get("gt_names", None)

    count = 0
    for i, box in enumerate(boxes[:max_gt]):
        corners = box_corners_bev(box)
        poly = Polygon(
            corners[:, :2],
            closed=True,
            fill=False,
            linestyle="--",
            linewidth=0.8,
        )
        ax.add_patch(poly)

        if names is not None and i < len(names):
            ax.text(float(box[0]), float(box[1]), str(names[i]), fontsize=5)

        count += 1

    return count


def save_language_guided_visualization(
    out_file: Path,
    mask: torch.Tensor,
    info: Dict,
    pred: Dict,
    instruction: str,
    metadata: Dict,
    score_thr: float,
    top_k: int,
) -> Dict[str, int]:
    """Save one language-guided BEV visualization image."""
    mask_np = mask.squeeze().detach().cpu().numpy()

    boxes = to_numpy(pred.get("boxes_3d", None))
    scores = to_numpy(pred.get("scores_3d", None))
    labels = to_numpy(pred.get("labels_3d", None))

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(
        mask_np,
        origin="lower",
        extent=[-50, 50, -50, 50],
        alpha=0.55,
    )

    gt_count = draw_gt_boxes(ax, info)
    pred_count = draw_boxes(
        ax,
        boxes=boxes,
        labels=labels,
        scores=scores,
        score_thr=score_thr,
        top_k=top_k,
        linestyle="-",
        linewidth=1.2,
    )

    ax.scatter([0], [0], marker="x", s=60)
    ax.text(0, 0, "ego", fontsize=8)

    token = info.get("token", "unknown")
    ax.set_title(
        "Language-Guided BEVFormer-tiny Visualization\n"
        f"Instruction: {instruction}\n"
        f"Sample: {token}"
    )
    ax.set_xlabel("x / forward direction")
    ax.set_ylabel("y / lateral direction")
    ax.set_xlim(-55, 55)
    ax.set_ylim(-55, 55)
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    fig.colorbar(im, ax=ax, label="language attention weight")
    fig.tight_layout()
    fig.savefig(out_file, dpi=180)
    plt.close(fig)

    return {
        "pred_count": pred_count,
        "gt_count": gt_count,
    }


def build_instruction_list() -> List[str]:
    """Return test instructions for V1 integration validation."""
    return [
        "focus on pedestrians in front of the ego vehicle",
        "pay attention to cars on the left",
        "watch nearby traffic cones",
        "focus on vehicles behind the ego car",
        "focus on distant trucks on the right",
        "pay attention to buses in front",
        "watch motorcycles on the left",
        "focus on nearby vehicles",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        default="work_dirs/bevformer_tiny_baseline/results_tiny.pkl",
    )
    parser.add_argument(
        "--infos",
        default="data/nuscenes/nuscenes_infos_temporal_val.pkl",
    )
    parser.add_argument(
        "--out-dir",
        default="../docs/visualizations/stage2_language_guided_tiny",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--score-thr", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=30)
    args = parser.parse_args()

    results_path = Path(args.results)
    infos_path = Path(args.infos)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        raise FileNotFoundError(f"Result file not found: {results_path}")

    if not infos_path.exists():
        raise FileNotFoundError(f"Info file not found: {infos_path}")

    print("Loading BEVFormer-tiny results:", results_path)
    outputs = mmcv.load(str(results_path))
    print("Number of prediction outputs:", len(outputs))

    print("Loading nuScenes val infos:", infos_path)
    with open(infos_path, "rb") as f:
        info_data = pickle.load(f)
    infos = info_data["infos"]
    print("Number of val infos:", len(infos))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = LanguageGuidedBEVModule(bev_h=200, bev_w=200, alpha=0.3).to(device)
    module.eval()

    instructions = build_instruction_list()
    num_samples = min(args.num_samples, len(outputs), len(infos), len(instructions))

    summary_lines = ["# Stage 2 V1 Language-Guided BEVFormer-tiny Integration Summary\n\n"]
    summary_lines.append(f"- Results file: `{results_path}`\n")
    summary_lines.append(f"- Info file: `{infos_path}`\n")
    summary_lines.append(f"- Number of processed samples: {num_samples}\n")
    summary_lines.append(f"- Score threshold: {args.score_thr}\n")
    summary_lines.append(f"- Top-K predictions: {args.top_k}\n\n")
    summary_lines.append(
        "| Index | Instruction | Parsed Target Classes | Spatial Hints | Pred Boxes | GT Boxes | Image |\n"
    )
    summary_lines.append("|---:|---|---|---|---:|---:|---|\n")

    bev_feature = torch.randn(1, 64, 200, 200, device=device)

    for idx in range(num_samples):
        instruction = instructions[idx]
        pred = get_pred_dict(outputs[idx])
        info = infos[idx]

        with torch.no_grad():
            _, mask, metadata = module(bev_feature, instruction)

        out_file = out_dir / f"language_guided_tiny_sample_{idx:04d}.png"
        stats = save_language_guided_visualization(
            out_file=out_file,
            mask=mask,
            info=info,
            pred=pred,
            instruction=instruction,
            metadata=metadata,
            score_thr=args.score_thr,
            top_k=args.top_k,
        )

        print("=" * 80)
        print("Index:", idx)
        print("Instruction:", instruction)
        print("Metadata:", metadata)
        print("Saved:", out_file)
        print("Pred boxes drawn:", stats["pred_count"])
        print("GT boxes drawn:", stats["gt_count"])

        summary_lines.append(
            f"| {idx} | `{instruction}` | `{metadata['target_classes']}` | "
            f"`{metadata['spatial_hints']}` | {stats['pred_count']} | "
            f"{stats['gt_count']} | `{out_file.name}` |\n"
        )

    summary_path = out_dir / "language_guided_tiny_integration_summary.md"
    summary_path.write_text("".join(summary_lines), encoding="utf-8")
    print("Summary saved:", summary_path)
    print("Stage 2 V1 language-guided BEVFormer-tiny integration finished.")


if __name__ == "__main__":
    main()
