import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "data" / "drivelm" / "QA_dataset_nus" / "v1_0_train_nus.json"
OUT_DIR = ROOT / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not JSON_PATH.exists():
    raise FileNotFoundError(f"Cannot find {JSON_PATH}")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

scene_count = len(data)
frame_count = 0
qa_type_counter = Counter()
qa_total_counter = Counter()
object_category_counter = Counter()
camera_counter = Counter()

examples = []

for scene_token, scene in data.items():
    key_frames = scene.get("key_frames", {})
    frame_count += len(key_frames)

    for frame_token, frame in key_frames.items():
        image_paths = frame.get("image_paths", {})
        for cam in image_paths.keys():
            camera_counter[cam] += 1

        key_objects = frame.get("key_object_infos", {})
        for obj_id, obj_info in key_objects.items():
            category = obj_info.get("Category", "UNKNOWN")
            object_category_counter[category] += 1

        qa_dict = frame.get("QA", {})
        for qa_type, qa_list in qa_dict.items():
            qa_type_counter[qa_type] += 1
            if isinstance(qa_list, list):
                qa_total_counter[qa_type] += len(qa_list)

        if len(examples) < 3:
            examples.append({
                "scene_token": scene_token,
                "scene_description": scene.get("scene_description", ""),
                "frame_token": frame_token,
                "image_paths": image_paths,
                "qa_types": list(qa_dict.keys()),
                "num_key_objects": len(key_objects),
            })

summary = {
    "json_path": str(JSON_PATH),
    "scene_count": scene_count,
    "frame_count": frame_count,
    "qa_type_counter": dict(qa_type_counter),
    "qa_total_counter": dict(qa_total_counter),
    "object_category_counter": dict(object_category_counter),
    "camera_counter": dict(camera_counter),
    "examples": examples,
}

out_path = OUT_DIR / "drivelm_summary.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("===== DriveLM Dataset Summary =====")
print("Scenes:", scene_count)
print("Frames:", frame_count)
print("QA totals:", dict(qa_total_counter))
print("Object categories:", dict(object_category_counter))
print("Cameras:", dict(camera_counter))
print("Saved to:", out_path)
