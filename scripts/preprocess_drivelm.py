import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "data" / "drivelm" / "QA_dataset_nus" / "v1_0_train_nus.json"
OUT_DIR = ROOT / "data" / "drivelm" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

frames_path = OUT_DIR / "drivelm_frames.jsonl"
qa_path = OUT_DIR / "drivelm_qa.jsonl"
objects_path = OUT_DIR / "drivelm_objects.jsonl"

if not JSON_PATH.exists():
    raise FileNotFoundError(f"Cannot find {JSON_PATH}")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

frame_count = 0
qa_count = 0
object_count = 0

with open(frames_path, "w", encoding="utf-8") as f_frames, \
     open(qa_path, "w", encoding="utf-8") as f_qa, \
     open(objects_path, "w", encoding="utf-8") as f_obj:

    for scene_token, scene in data.items():
        scene_description = scene.get("scene_description", "")
        key_frames = scene.get("key_frames", {})

        for frame_token, frame in key_frames.items():
            image_paths = frame.get("image_paths", {})
            key_objects = frame.get("key_object_infos", {})
            qa_dict = frame.get("QA", {})

            frame_record = {
                "scene_token": scene_token,
                "frame_token": frame_token,
                "scene_description": scene_description,
                "image_paths": image_paths,
                "num_key_objects": len(key_objects),
                "qa_types": list(qa_dict.keys()),
            }
            f_frames.write(json.dumps(frame_record, ensure_ascii=False) + "\n")
            frame_count += 1

            for object_id, object_info in key_objects.items():
                obj_record = {
                    "scene_token": scene_token,
                    "frame_token": frame_token,
                    "object_id": object_id,
                    "category": object_info.get("Category"),
                    "status": object_info.get("Status"),
                    "visual_description": object_info.get("Visual_description"),
                    "bbox_2d": object_info.get("2d_bbox"),
                    "raw": object_info,
                }
                f_obj.write(json.dumps(obj_record, ensure_ascii=False) + "\n")
                object_count += 1

            for qa_type, qa_list in qa_dict.items():
                if not isinstance(qa_list, list):
                    continue

                for idx, qa_item in enumerate(qa_list):
                    qa_record = {
                        "scene_token": scene_token,
                        "frame_token": frame_token,
                        "qa_type": qa_type,
                        "qa_index": idx,
                        "question": qa_item.get("Q"),
                        "answer": qa_item.get("A"),
                        "chain": qa_item.get("C"),
                        "con_up": qa_item.get("con_up"),
                        "con_down": qa_item.get("con_down"),
                        "cluster": qa_item.get("cluster"),
                        "layer": qa_item.get("layer"),
                    }
                    f_qa.write(json.dumps(qa_record, ensure_ascii=False) + "\n")
                    qa_count += 1

print("Preprocessing completed.")
print("Frames:", frame_count, "->", frames_path)
print("QA pairs:", qa_count, "->", qa_path)
print("Objects:", object_count, "->", objects_path)
