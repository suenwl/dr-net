# from pdf2image import convert_from_path
import fitz
from PIL import ImageDraw
from PIL import Image
from OCREngine import OCREngine
from Token import Token
from util import convert_pdf_to_image
import re

import cv2
import numpy as np


class Invoice:
    def __init__(self, PDF_path: str):
        self.pages = [
            InvoicePage(page) for page in convert_pdf_to_image(PDF_path)
        ]  # Each of the individual pages in the PDF is converted to images

        self.is_text_based = (
            False
        )  # TODO: Need to implement way to check if PDF is text based

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


class InvoicePage:
    def __init__(self, image: Image):
        self.page = image
        self.tokens = None
        self.regions = None
        self.tokens_by_block = None
        self.tokens_no_stopwords = None

    def do_OCR(self):
        if not self.tokens:
            ocr_engine = OCREngine()
            self.tokens, self.regions = ocr_engine.OCR(self.page)

    def get_tokens_by_block(self, block_num: int = None):
        self.do_OCR()
        if self.tokens_by_block:
            return self.tokens_by_block

        blocks = {}

        for token in self.tokens:
            block_num = token.token_structure["block_num"]
            if block_num in blocks:
                blocks[block_num].append(token)
            else:
                blocks[block_num] = [token]

        self.tokens_by_block = blocks

        return blocks

    def search_tokens(self, text: str):
        self.do_OCR()
        filtered_tokens = list(
            filter(lambda token: bool(re.search(text, token.text.lower())), self.tokens)
        )

        return filtered_tokens

    def draw_bounding_boxes(
        self, detail="block"
    ):  # detail can be block, paragraph, line, word
        def draw_rect(canvas: ImageDraw, token: Token, colour: tuple):
            canvas.rectangle(
                (
                    token.coordinates["x"],
                    token.coordinates["y"],
                    token.coordinates["x"] + token.coordinates["width"],
                    token.coordinates["y"] + token.coordinates["height"],
                ),
                outline=colour,
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
        else:
            raise Exception(
                "Invalid option for detail selected. Can only be 'block', 'paragraph', 'line', or 'word'"
            )

        for token in selected_to_draw:
            draw_rect(canvas, token, (255, 0, 0))

        page_copy.show()

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


##### TODO: The following code is relevant to text-based invoices and needs to be integrated
##### into the invoice class in the future


def convert_text_to_result(text):

    result = (
        None
    )  # TODO: Use invoice2data or other means to obtain results using text from invoice

    return result


def convert_text_based_pdf_to_result(invoice):
    templates = read_templates("./templates")

    # TODO: Implement data extraction using invoice2data

    return None
