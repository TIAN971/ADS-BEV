import argparse
import math
import pickle
from pathlib import Path

import mmcv
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


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
    """Convert tensors / mmdet3d boxes / lists to numpy."""
    if x is None:
        return None
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "cpu"):
        x = x.cpu().numpy()
    else:
        x = np.asarray(x)
    return x


def get_pred_dict(result_item):
    """Handle BEVFormer output format."""
    if isinstance(result_item, dict) and "pts_bbox" in result_item:
        return result_item["pts_bbox"]
    return result_item


def box_corners_bev(box):
    """
    box format assumed:
    [x, y, z, dx, dy, dz, yaw, ...]
    x/y are BEV center coordinates.
    dx/dy are box size on BEV plane.
    yaw is rotation angle.
    """
    x, y = float(box[0]), float(box[1])
    dx, dy = float(box[3]), float(box[4])
    yaw = float(box[6]) if len(box) > 6 else 0.0

    local = np.array([
        [ dx / 2,  dy / 2],
        [ dx / 2, -dy / 2],
        [-dx / 2, -dy / 2],
        [-dx / 2,  dy / 2],
    ])

    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    corners = local @ rot.T
    corners[:, 0] += x
    corners[:, 1] += y
    return corners


def draw_boxes(ax, boxes, labels=None, scores=None, score_thr=0.0, top_k=50, linestyle="-", title_prefix="Pred"):
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

        poly = Polygon(corners[:, :2], closed=True, fill=False, linestyle=linestyle, linewidth=1.2)
        ax.add_patch(poly)

        x, y = float(box[0]), float(box[1])
        label_text = title_prefix
        if labels is not None:
            label_id = int(labels[idx]) if idx < len(labels) else -1
            if 0 <= label_id < len(NUSCENES_CLASSES):
                label_text = NUSCENES_CLASSES[label_id]
        if scores is not None:
            label_text += f" {float(scores[idx]):.2f}"

        ax.text(x, y, label_text, fontsize=6)
        count += 1

    return count


def draw_gt_boxes(ax, info, max_gt=80):
    if "gt_boxes" not in info:
        return 0

    boxes = np.asarray(info["gt_boxes"])
    names = info.get("gt_names", None)

    count = 0
    for i, box in enumerate(boxes[:max_gt]):
        corners = box_corners_bev(box)
        poly = Polygon(corners[:, :2], closed=True, fill=False, linestyle="--", linewidth=0.8)
        ax.add_patch(poly)

        if names is not None and i < len(names):
            ax.text(float(box[0]), float(box[1]), str(names[i]), fontsize=5)

        count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="work_dirs/bevformer_tiny_baseline/results_tiny.pkl")
    parser.add_argument("--infos", default="data/nuscenes/nuscenes_infos_temporal_val.pkl")
    parser.add_argument("--out-dir", default="../docs/visualizations/bevformer_tiny")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--score-thr", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    results_path = Path(args.results)
    infos_path = Path(args.infos)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results:", results_path)
    outputs = mmcv.load(str(results_path))
    print("Number of prediction outputs:", len(outputs))

    print("Loading infos:", infos_path)
    with open(infos_path, "rb") as f:
        info_data = pickle.load(f)
    infos = info_data["infos"]
    print("Number of val infos:", len(infos))

    summary_lines = []
    summary_lines.append("# BEVFormer-tiny BEV Visualization Summary\n")
    summary_lines.append(f"- Results file: `{results_path}`\n")
    summary_lines.append(f"- Info file: `{infos_path}`\n")
    summary_lines.append(f"- Total prediction outputs: {len(outputs)}\n")
    summary_lines.append(f"- Total val infos: {len(infos)}\n")
    summary_lines.append(f"- Score threshold: {args.score_thr}\n")
    summary_lines.append(f"- Top-K predictions per sample: {args.top_k}\n\n")
    summary_lines.append("| Index | Sample Token | Pred Boxes Drawn | GT Boxes Drawn | Image |\n")
    summary_lines.append("|---:|---|---:|---:|---|\n")

    end = min(args.start_index + args.num_samples, len(outputs), len(infos))

    for idx in range(args.start_index, end):
        pred = get_pred_dict(outputs[idx])
        info = infos[idx]

        boxes = to_numpy(pred.get("boxes_3d", None)) if isinstance(pred, dict) else None
        scores = to_numpy(pred.get("scores_3d", None)) if isinstance(pred, dict) else None
        labels = to_numpy(pred.get("labels_3d", None)) if isinstance(pred, dict) else None

        fig, ax = plt.subplots(figsize=(8, 8))

        # Ego vehicle position
        ax.scatter([0], [0], marker="x", s=60)
        ax.text(0, 0, "ego", fontsize=8)

        gt_count = draw_gt_boxes(ax, info)
        pred_count = draw_boxes(
            ax,
            boxes,
            labels=labels,
            scores=scores,
            score_thr=args.score_thr,
            top_k=args.top_k,
            linestyle="-",
            title_prefix="Pred",
        )

        token = info.get("token", f"idx_{idx}")
        ax.set_title(f"BEVFormer-tiny BEV Predictions | sample {idx}\n{token}")
        ax.set_xlabel("x / forward direction")
        ax.set_ylabel("y / lateral direction")
        ax.set_xlim(-55, 55)
        ax.set_ylim(-55, 55)
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")

        out_file = out_dir / f"bev_tiny_sample_{idx:04d}.png"
        fig.tight_layout()
        fig.savefig(out_file, dpi=180)
        plt.close(fig)

        print(f"Saved {out_file} | pred boxes: {pred_count}, gt boxes: {gt_count}")

        summary_lines.append(
            f"| {idx} | `{token}` | {pred_count} | {gt_count} | `{out_file.name}` |\n"
        )

    summary_path = out_dir / "visualization_summary.md"
    summary_path.write_text("".join(summary_lines), encoding="utf-8")
    print("Summary saved to:", summary_path)


if __name__ == "__main__":
    main()
