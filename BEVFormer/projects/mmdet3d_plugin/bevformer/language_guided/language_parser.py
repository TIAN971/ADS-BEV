from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ParsedInstruction:
    """Parsed result of a natural language driving instruction.

    Attributes:
        raw_text: Original instruction text.
        target_classes: Object classes mentioned by the instruction.
        spatial_hints: Spatial hints such as front, rear, left, right, near.
        priority: Attention strength level.
    """

    raw_text: str
    target_classes: List[str]
    spatial_hints: List[str]
    priority: float


class RuleBasedLanguageParser:
    """A lightweight rule-based parser for driving instructions.

    This parser provides a stable functional baseline for Stage 2. It extracts
    target object classes and spatial hints from natural language instructions.
    The parser can later be replaced by a pretrained language encoder while
    keeping the same output interface.
    """

    def __init__(self) -> None:
        self.class_keywords: Dict[str, List[str]] = {
            "car": ["car", "vehicle", "cars", "vehicles"],
            "truck": ["truck", "trucks"],
            "bus": ["bus", "buses"],
            "pedestrian": ["pedestrian", "pedestrians", "person", "people"],
            "bicycle": ["bicycle", "bike", "cyclist"],
            "motorcycle": ["motorcycle", "motorbike"],
            "traffic_cone": ["traffic cone", "cone", "cones"],
            "barrier": ["barrier", "barriers"],
        }

        self.spatial_keywords: Dict[str, List[str]] = {
            "front": ["front", "ahead", "forward", "in front"],
            "rear": ["rear", "behind", "back"],
            "left": ["left"],
            "right": ["right"],
            "near": ["near", "nearby", "close"],
            "far": ["far", "distant"],
        }

    def parse(self, instruction: str) -> ParsedInstruction:
        """Parse an instruction into target classes and spatial hints.

        Args:
            instruction: Natural language driving instruction.

        Returns:
            ParsedInstruction containing target classes and spatial hints.
        """
        if not isinstance(instruction, str) or len(instruction.strip()) == 0:
            raise ValueError("instruction must be a non-empty string")

        text = instruction.lower()
        target_classes: List[str] = []
        spatial_hints: List[str] = []

        for class_name, keywords in self.class_keywords.items():
            if any(keyword in text for keyword in keywords):
                target_classes.append(class_name)

        for hint, keywords in self.spatial_keywords.items():
            if any(keyword in text for keyword in keywords):
                spatial_hints.append(hint)

        if not target_classes:
            target_classes = ["car", "pedestrian"]

        if not spatial_hints:
            spatial_hints = ["front"]

        priority = 1.0
        if any(word in text for word in ["important", "focus", "attention", "careful"]):
            priority = 1.5

        return ParsedInstruction(
            raw_text=instruction,
            target_classes=target_classes,
            spatial_hints=spatial_hints,
            priority=priority,
        )
