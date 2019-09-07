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

    def __repr__(self):
        return self.text if self.text else str(self.token_structure)

    def __str__(self):
        return self.text if self.text else str(self.token_structure)

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


# def convert_ocr_dataframe_to_token_list(ocr_dataframe):

#     token_list = []

#     for index, row in ocr_dataframe.iterrows():

#         token_list.append(
#             Token(
#                 row.text,
#                 {"x": row.left, "y": row.top, "height": row.height, "width": row.width},
#                 row.conf,
#                 {
#                     "page_num": row.page_num,
#                     "block_num": row.block_num,
#                     "par_num": row.par_num,
#                     "line_num": row.line_num,
#                     "word_num": row.word_num,
#                 },
#             )
#         )

#     return token_list
