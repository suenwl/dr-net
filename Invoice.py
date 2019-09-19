# from pdf2image import convert_from_path
import fitz
from PIL import ImageDraw
from PIL import Image
from OCREngine import OCREngine
from Token import Token
from util import convert_pdf_to_image
import json
import re

import cv2
import numpy as np


class Invoice:
    def __init__(self, PDF_path: str):
        self.readable_name = PDF_path.split("/")[-1]
        self.original_file_path = PDF_path
        self.pages = [
            InvoicePage(page) for page in convert_pdf_to_image(PDF_path)
        ]  # Each of the individual pages in the PDF is converted to images

    def length(self):
        return len(self.pages)

    def get_all_tokens(self):
        ocr_engine = OCREngine()
        return {
            page_number + 1: ocr_engine.OCR(page)
            for (page_number, page) in enumerate(self.pages)
        }

    def get_page(self, page_number: int):
        return self.pages[page_number - 1]

    def map_labels(self, json_file_path="", verbose=False):
        """Maps json labels, created from the pdf labeller, to the existing grouped tokens in the invoice"""
        if not json_file_path:
            json_file_path = self.original_file_path[:-4] + ".json"
        try:
            if verbose:
                print("Loading labels from", json_file_path)
            categories = json.load(open(json_file_path, "r"))
        except IOError:
            print(
                "WARNING: json tags for the PDF at",
                self.original_file_path,
                "does not exist. Check if the path provided was correct. Skipping this pdf",
            )

        # Process all the labels
        for category in categories:
            category_label = category["category"]

            for label in category["items"]:
                coordinates = {
                    k: v for k, v in label.items() if k in ["x", "y", "width", "height"]
                }
                page_number = label["page"]
                page = self.get_page(page_number)
                token_to_label = page.find_overlapping_token(coordinates)
                if verbose:
                    print(
                        "FOUND TOKEN",
                        token_to_label,
                        "Setting category as",
                        category_label,
                    )
                token_to_label.set_category(category_label)


class InvoicePage:
    def __init__(self, image: Image):
        self.page = image
        self.tokens = None
        self.grouped_tokens = None
        self.regions = None
        self.tokens_by_block_and_line = None

    def do_OCR(self, verbose=False):
        if not self.tokens:
            ocr_engine = OCREngine()
            self.tokens, self.grouped_tokens, self.tokens_by_block_and_line, self.regions = ocr_engine.OCR(
                self.page, verbose=verbose
            )

    def search_tokens(self, text: str, token_list="group"):
        self.do_OCR()
        if token_list == "group":
            token_list_to_search = self.grouped_tokens
        elif token_list == "word":
            token_list_to_search = self.tokens
        filtered_tokens = list(
            filter(
                lambda token: bool(re.search(text, token.text.lower())),
                token_list_to_search,
            )
        )

        return filtered_tokens

    def find_overlapping_token(self, coordinates):
        THRESHOLD = 0.5
        max_overlap = 0
        for token in self.grouped_tokens:
            percentage_overlap = token.get_percentage_overlap(
                coordinates, self.page.size
            )
            max_overlap = max(max_overlap, percentage_overlap)
            if percentage_overlap >= THRESHOLD:
                return token
        raise Exception(
            "No significant overlap between token and label at",
            coordinates,
            "was found. Maximum overlap was",
            max_overlap,
        )

    def draw_bounding_boxes(
        self, detail="block"
    ):  # detail can be block, paragraph, line, word
        def draw_rect(canvas: ImageDraw, token: Token, colour: tuple, width: int = 1):
            canvas.rectangle(
                (
                    token.coordinates["x"],
                    token.coordinates["y"],
                    token.coordinates["x"] + token.coordinates["width"],
                    token.coordinates["y"] + token.coordinates["height"],
                ),
                outline=colour,
                width=width,
            )

        page_copy = self.page.copy()
        canvas = ImageDraw.Draw(page_copy)
        if detail == "block":
            selected_to_draw = list(
                filter(
                    lambda region: region.token_structure["par_num"] == 0, self.regions
                )
            )
        elif detail == "paragraph":
            selected_to_draw = list(
                filter(
                    lambda region: region.token_structure["par_num"] != 0
                    and region.token_structure["line_num"] == 0,
                    self.regions,
                )
            )
        elif detail == "line":
            selected_to_draw = list(
                filter(
                    lambda region: region.token_structure["par_num"] != 0
                    and region.token_structure["line_num"] != 0
                    and region.token_structure["word_num"] == 0,
                    self.regions,
                )
            )
        elif detail == "word":
            selected_to_draw = self.tokens

        elif detail == "group":
            selected_to_draw = self.grouped_tokens
        else:
            raise Exception(
                "Invalid option for detail selected. Can only be 'block', 'paragraph', 'group', 'line', or 'word'"
            )

        for token in selected_to_draw:
            if token.category:  # Emphasise if this token has been labelled
                draw_rect(canvas, token, (255, 0, 0), 3)
            else:
                draw_rect(canvas, token, (0, 255, 0))

        page_copy.show()

    # Serialiser
    def serialise(self):
        return {
            k: v for k, v in self.__dict__.items() if k != "page"
        }  # Do not include the page, since it is an image

    # Save output by extracting text from token objects for NLP experimentation
    def write_output_json(self, fileName):
        newdict = {
            k: list(map(lambda x: x.text, v)) for k, v in self.tokens_by_block.items()
        }
        with open(fileName, "w") as f:
            json.dump(newdict, f, ensure_ascii=False)

    def remove_lines(self):
        pil_image = self.page.convert("RGB")
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        img = open_cv_image[:, :, ::-1].copy()
        # Convert to grey for easier processing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # inverse colour og image for easier processing of lines
        img = cv2.bitwise_not(img)
        th2 = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2
        )
        # show blackwhite version
        # cv2.imshow("blackwhite_orginal", th2)
        # cv2.imwrite("blackwhite_orginal.jpg", th2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        horizontal = th2
        vertical = th2
        rows, cols = horizontal.shape
        horizontalsize = int(cols / 15)
        horizontalStructure = cv2.getStructuringElement(
            cv2.MORPH_RECT, (horizontalsize, 1)
        )
        horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
        horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
        # show horizontal lines
        # cv2.imshow("horizontal", horizontal)
        # cv2.imwrite("horizontal.jpg", horizontal)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print('horizontal')

        # inverse the image, so that lines are black for masking
        horizontal_inv = cv2.bitwise_not(horizontal)
        # perform bitwise_and to mask the lines with provided mask
        masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
        # reverse the image back to normal
        masked_img_inv = cv2.bitwise_not(masked_img)
        # show removal of horizontal lines
        # cv2.imshow("masked img", masked_img_inv)
        # cv2.imwrite("result2.jpg", masked_img_inv)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print('masked_img_inv')

        verticalsize = int(rows / 30)
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
        vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
        # show vertical lines
        # cv2.imshow("vertical", vertical)
        # cv2.imwrite("vertical.jpg", vertical)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print('vertical')

        masked_img_inv = cv2.bitwise_not(masked_img_inv)
        # inverse the image, so that lines are black for masking
        vertical_inv = cv2.bitwise_not(vertical)
        # perform bitwise_and to mask the lines with provided mask
        masked_img2 = cv2.bitwise_and(masked_img_inv, masked_img_inv, mask=vertical_inv)
        # reverse the image back to normal
        masked_img_inv2 = cv2.bitwise_not(masked_img2)
        # show final result
        # cv2.imshow("masked img2", masked_img_inv2)
        cv2.imwrite("final_result.jpg", masked_img_inv2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        self.page = masked_img_inv2
