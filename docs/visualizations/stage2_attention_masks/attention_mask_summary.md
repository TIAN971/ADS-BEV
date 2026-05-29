# Stage 2 BEV Attention Mask Visualization

## Sample 0

- Instruction: `focus on pedestrians in front of the ego vehicle`
- Parsed metadata: `{'raw_text': 'focus on pedestrians in front of the ego vehicle', 'target_classes': ['car', 'pedestrian'], 'spatial_hints': ['front'], 'priority': 1.5}`
- Output image: `attention_mask_00.png`

## Sample 1

- Instruction: `pay attention to cars on the left`
- Parsed metadata: `{'raw_text': 'pay attention to cars on the left', 'target_classes': ['car'], 'spatial_hints': ['left'], 'priority': 1.5}`
- Output image: `attention_mask_01.png`

## Sample 2

- Instruction: `watch nearby traffic cones`
- Parsed metadata: `{'raw_text': 'watch nearby traffic cones', 'target_classes': ['traffic_cone'], 'spatial_hints': ['near'], 'priority': 1.0}`
- Output image: `attention_mask_02.png`

## Sample 3

- Instruction: `focus on vehicles behind the ego car`
- Parsed metadata: `{'raw_text': 'focus on vehicles behind the ego car', 'target_classes': ['car'], 'spatial_hints': ['rear'], 'priority': 1.5}`
- Output image: `attention_mask_03.png`

## Sample 4

- Instruction: `focus on distant trucks on the right`
- Parsed metadata: `{'raw_text': 'focus on distant trucks on the right', 'target_classes': ['truck'], 'spatial_hints': ['right', 'far'], 'priority': 1.5}`
- Output image: `attention_mask_04.png`

