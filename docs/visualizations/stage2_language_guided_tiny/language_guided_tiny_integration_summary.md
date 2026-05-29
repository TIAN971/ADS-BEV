# Stage 2 V1 Language-Guided BEVFormer-tiny Integration Summary

- Results file: `work_dirs/bevformer_tiny_baseline/results_tiny.pkl`
- Info file: `data/nuscenes/nuscenes_infos_temporal_val.pkl`
- Number of processed samples: 8
- Score threshold: 0.2
- Top-K predictions: 30

| Index | Instruction | Parsed Target Classes | Spatial Hints | Pred Boxes | GT Boxes | Image |
|---:|---|---|---|---:|---:|---|
| 0 | `focus on pedestrians in front of the ego vehicle` | `['car', 'pedestrian']` | `['front']` | 30 | 37 | `language_guided_tiny_sample_0000.png` |
| 1 | `pay attention to cars on the left` | `['car']` | `['left']` | 20 | 38 | `language_guided_tiny_sample_0001.png` |
| 2 | `watch nearby traffic cones` | `['traffic_cone']` | `['near']` | 27 | 40 | `language_guided_tiny_sample_0002.png` |
| 3 | `focus on vehicles behind the ego car` | `['car']` | `['rear']` | 30 | 41 | `language_guided_tiny_sample_0003.png` |
| 4 | `focus on distant trucks on the right` | `['truck']` | `['right', 'far']` | 30 | 42 | `language_guided_tiny_sample_0004.png` |
| 5 | `pay attention to buses in front` | `['bus']` | `['front']` | 24 | 42 | `language_guided_tiny_sample_0005.png` |
| 6 | `watch motorcycles on the left` | `['motorcycle']` | `['left']` | 30 | 42 | `language_guided_tiny_sample_0006.png` |
| 7 | `focus on nearby vehicles` | `['car']` | `['near']` | 30 | 44 | `language_guided_tiny_sample_0007.png` |
