# OCR toolkit
import pytesseract
import pandas as pd
from PIL import Image
from typing import Dict
from math import sqrt


class Token:
    def __init__(
        self,
        text: str,
        coordinates: Dict[str, int],
        confidence: int,
        token_structure: Dict[str, int],
    ):
        self.text = text
        self.coordinates = coordinates
        self.confidence = confidence
        self.token_structure = token_structure
        self.category = None

    def __repr__(self):
        return self.text if self.text else str(self.token_structure)

    def __str__(self):
        return self.text if self.text else str(self.token_structure)

    def set_category(self, category: str):
        self.category = category

    def is_horizontally_aligned_with(self, token):
        token_vertical_midpoint = (
            token.coordinates["y"] + token.coordinates["height"] / 2
        )
        return (
            self.coordinates["y"] < token_vertical_midpoint
            and token_vertical_midpoint
            < self.coordinates["y"] + self.coordinates["height"]
        )

    def is_vertically_aligned_with(self, token):
        token_horizontal_midpoint = (
            token.coordinates["x"] + token.coordinates["width"] / 2
        )
        return (
            self.coordinates["x"]
            < token_horizontal_midpoint
            < self.coordinates["x"] + self.coordinates["width"]
        )

    def is_aligned_with(self, token):
        return self.is_horizontally_aligned_with(
            token
        ) or self.is_vertically_aligned_with(token)

    def get_distance_to(self, token):
        token_vertical_midpoint = (
            token.coordinates["y"] + token.coordinates["height"] / 2
        )
        token_horizontal_midpoint = (
            token.coordinates["x"] + token.coordinates["width"] / 2
        )

        self_vertical_midpoint = self.coordinates["y"] + self.coordinates["height"] / 2
        self_horizontal_midpoint = self.coordinates["x"] + self.coordinates["width"] / 2

        return sqrt(
            (self_vertical_midpoint - token_vertical_midpoint) ** 2
            + (self_horizontal_midpoint - token_horizontal_midpoint) ** 2
        )

    def get_aligned_tokens(self, token_list, option="all"):
        if option == "all":
            return [
                token
                for token in token_list
                if self.is_horizontally_aligned_with(token)
                or self.is_vertically_aligned_with(token)
            ]
        elif option == "vertical":
            return [
                token for token in token_list if self.is_vertically_aligned_with(token)
            ]
        elif option == "horizontal":
            return [
                token
                for token in token_list
                if self.is_horizontally_aligned_with(token)
            ]
        else:
            raise Exception("Invalid option provided for get_aligned_tokens")

    def get_percentage_overlap(self, coordinates, image_size):
        image_width = image_size[0]
        image_height = image_size[1]

        token_x1 = self.coordinates["x"] / image_width
        token_x2 = token_x1 + self.coordinates["width"] / image_width
        token_y1 = self.coordinates["y"] / image_height
        token_y2 = token_y1 + self.coordinates["height"] / image_height
        token_area = abs(token_x1 - token_x2) * abs(token_y1 - token_y2)

        rect_x1 = coordinates["x"]
        rect_x2 = rect_x1 + coordinates["width"]
        rect_y1 = coordinates["y"]
        rect_y2 = rect_y1 + coordinates["height"]
        rect_area = abs(rect_x1 - rect_x2) * abs(rect_y1 - rect_y2)

        x1 = max(min(token_x1, token_x2), min(rect_x1, rect_x2))
        y1 = max(min(token_y1, token_y2), min(rect_y1, rect_y2))
        x2 = min(max(token_x1, token_x2), max(rect_x1, rect_x2))
        y2 = min(max(token_y1, token_y2), max(rect_y1, rect_y2))
        if x1 < x2 and y1 < y2:  # If there is an overlap
            overlap_area = abs(x2 - x1) * abs(y2 - y1)
            return overlap_area / (rect_area + token_area - overlap_area)
        else:
            return 0
