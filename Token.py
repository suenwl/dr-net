# OCR toolkit
import pytesseract
import pandas as pd
from PIL import Image
from typing import Dict
from math import sqrt
import re


class Token:
    def __init__(
        self,
        text: str,
        coordinates: Dict[str, int],
        confidence: int,
        token_structure: Dict[str, int],
        category: str = None,
    ):
        self.text = text
        self.coordinates = coordinates
        self.confidence = confidence
        self.token_structure = token_structure
        self.category = category

        # feature related fields
        self.date_values = self.get_dates()
        self.currency = self.get_currency()
        self.address = self.get_address()
        self.num_label = self.get_num_label()
        self.total_label = self.get_total_label()
        self.date_label = self.get_date_label()
        self.contains_digit = self.get_contains_digits()

    def __repr__(self):
        return self.text if self.text else str(self.token_structure)

    def __str__(self):
        return self.text if self.text else str(self.token_structure)

    # returns true if has at least one digit, false otherwise
    def get_contains_digits(self):
        if self.text:
            for c in self.text:
                if c.isdigit():
                    return True
            return False

    def get_date_label(self):
        kw = ["date", "Date"]
        if self.text:
            for w in kw:
                if w in self.text:
                    return self.text

    # tries to extract address from token
    def get_address(self):
        kw = [
            "drive",
            "dr",
            "road",
            "rd",
            "lane",
            "ln",
            "ave",
            "avenue",
            "street",
            "jalan",
            "jln",
            "boulevard",
            "blvd",
            "way",
            "park",
            "gate",
            "crescent",
            "cres",
            "singapore",
            "grove",
            "grv",
            "lorong",
            "lor",
            "valley",
            "view",
            "bukit",
            "tanjong",
        ]
        if self.text:
            text_array = self.text.lower().split(" ")
            for w in kw:
                for t in text_array:
                    if re.search("^" + w, t) and len(text_array) < 10:
                        return self.text

    # returns the text if "total" or some variant is contained in text and group is fewer than 5 words
    def get_total_label(self):
        kw = ["total", "Total"]
        if self.text:
            for w in kw:
                if w in self.text and len(self.text.split(" ")) < 5:
                    return self.text

    # returns string for description of number, eg. account number, invoice number
    def get_num_label(self):
        kw = ["no.", "no:", "no.:", "number", "num", "#", "#:"]
        if self.text:
            text_array = self.text.lower().split(" ")
            for w in kw:
                if w in text_array and text_array.index(w) > 0:
                    return self.text

    # returns a dictionary of {cur: <prefix> , value: <dollar amt> }
    # eg. {cur: $ , value: 5.00 }
    def get_currency(self):
        currencies = ["$", "¥", "dollar", "SGD", "USD", "US$", "SG$", "$SG", "$US"]
        out = {}
        for cur in currencies:
            if self.text and cur in self.text:
                out["cur"] = cur
                start = self.text.index(cur) + len(cur)
                for i in range(start, len(self.text)):
                    char = self.text[i]
                    if not char.isdigit() and not (char in [".", ","]):
                        break
                try:
                    out["val"] = float(self.text[start : i + 1])
                    return out
                except:
                    return None

    # checks if token is a date token
    def get_dates(self):
        # TODO parse range of dates and into date objects
        dates = []
        text = self.text
        if type(text) is str:
            month_names = set(
                [
                    "jan",
                    "feb",
                    "mar",
                    "apr",
                    "may",
                    "jun",
                    "jul",
                    "aug",
                    "sep",
                    "oct",
                    "nov",
                    "dec",
                ]
            )

            # checks for numerical months using regex
            text_nospaces = text.replace(" ", "")
            # matches d/dd or d-dd
            re_date = re.search("\d[/|-]\d\d", text_nospaces)
            if re_date:
                dates.append(text_nospaces)

            text_list = text.split(" ")

            # if token has multiple words
            if len(text_list) > 1:
                # checks for named months
                for index, word in enumerate(text_list):
                    for month in month_names:
                        if month in word.lower():
                            date = []
                            if index > 0:
                                date.append(text_list[index - 1])

                            date.append(text_list[index])

                            if index < len(text_list) - 1:
                                date.append(text_list[index + 1])

                            if len(date) > 1:
                                dates.append(" ".join(date))

            # token has only one word
            else:
                for month in month_names:
                    if month in text_list[0]:
                        dates.append(text_list[0])
        return dates

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
