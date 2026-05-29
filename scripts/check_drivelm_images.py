import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMES_PATH = ROOT / "data" / "drivelm" / "processed" / "drivelm_frames.jsonl"
NUSC_ROOT = ROOT / "data" / "nuscenes"

def resolve_image_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p

    candidates = [
        NUSC_ROOT / p,
        ROOT / "data" / "drivelm" / p,
        ROOT / p,
    ]

    for c in candidates:
        if c.exists():
            return c

    return candidates[0]

total = 0
missing = 0
missing_examples = []

if not FRAMES_PATH.exists():
    raise FileNotFoundError(f"Cannot find {FRAMES_PATH}. Run preprocess_drivelm.py first.")

with open(FRAMES_PATH, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        for cam, path_str in row.get("image_paths", {}).items():
            total += 1
            real_path = resolve_image_path(path_str)
            if not real_path.exists():
                missing += 1
                if len(missing_examples) < 20:
                    missing_examples.append({
                        "frame_token": row["frame_token"],
                        "camera": cam,
                        "path": path_str,
                        "resolved": str(real_path),
                    })

print("Total image references:", total)
print("Missing images:", missing)

if missing_examples:
    print("Missing examples:")
    for item in missing_examples:
        print(item)
else:
    print("All checked image paths exist.")
