# from pdf2image import convert_from_path
import fitz
from PIL import ImageDraw
from PIL import Image
from OCREngine import OCREngine
from Token import Token
from util import convert_pdf_to_image
import re
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords


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

    def remove_stopwords(self):
        if self.tokens:
            stopwords_set = set(stopwords.words("english"))
            self.tokens_no_stopwords = list(
                filter(lambda t: t.text not in stopwords_set, self.tokens)
            )

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
