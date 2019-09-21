# from pdf2image import convert_from_path
import fitz
from PIL import ImageDraw, Image, ImageFont
from OCREngine import OCREngine
from Token import Token
from util import convert_pdf_to_image
import json
import re


class Invoice:
    def __init__(self, PDF_path: str):
        self.readable_name = PDF_path.split("/")[-1]
        self.original_file_path = PDF_path
        self.pages = [
            InvoicePage(page) for page in convert_pdf_to_image(PDF_path)
        ]  # Each of the individual pages in the PDF is converted to images

    def length(self):
        return len(self.pages)

    def save_data(self, file_name: str = None):
        """Save all of this invoice's data (OCR outputs) into a json save file"""
        file_path_stem = self.original_file_path.rsplit("/", 1)[0] + "/"
        if not file_name:
            file_path = file_path_stem + self.readable_name[:-4] + "-savefile.json"
        else:
            file_path = file_path_stem + file_name

        with open(file_path, "w") as save_file:
            json.dump(self, save_file, cls=ObjectEncoder)

    def load_data(self, file_name: str = None):
        """Loads invoice save data from a json save file, given the file name"""
        file_path_stem = self.original_file_path.rsplit("/", 1)[0] + "/"
        if not file_name:
            file_path = file_path_stem + self.readable_name[:-4] + "-savefile.json"

        with open(file_path, "r") as save_file:
            data = json.load(save_file)
            for i, page_data in enumerate(data["pages"]):
                page = self.pages[i]
                page.load_data(page_data)

    def get_all_tokens(self):
        ocr_engine = OCREngine()
        return {
            page_number + 1: ocr_engine.OCR(page)
            for (page_number, page) in enumerate(self.pages)
        }

    def do_OCR(self, range: tuple = None, verbose: bool = False):
        """Performs OCR on the entire invoice. A range of pages can be provided"""
        if not range:
            range = (0, self.length())
        for page in self.pages[range[0] : range[1]]:
            page.do_OCR(verbose=verbose)

    def get_page(self, page_number: int):
        return self.pages[page_number - 1]

    def map_labels(self, json_file_path="", verbose: bool = False):
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
        self.processed_page = OCREngine.preprocess_image(image)
        self.tokens = None
        self.grouped_tokens = None
        self.regions = None
        self.tokens_by_block_and_line = None

    def load_data(self, data_packet: dict):
        """Loads tokens, grouped_tokens, regions, tokens_by_block_and_line using a data packet. Raises an error if data is already populated"""
        existing_data = [
            self.tokens,
            self.grouped_tokens,
            self.regions,
            self.tokens_by_block_and_line,
        ]

        if any(existing_data):  # If any of the data already exists in the invoicePage
            raise Exception(
                "InvoicePage data loading error: Data already exists in InvoicePage object. Data can only be loaded onto a fresh InvoicePage"
            )

        if not all(
            [data for key, data in data_packet.items()]
        ):  # If not all data in data_packet is present
            return  # We probably didn't do OCR for this page previously, so just return

        create_tokens_from_dict = lambda dictionary: Token(**dictionary)

        self.tokens = list(map(create_tokens_from_dict, data_packet["tokens"]))
        self.grouped_tokens = list(
            map(create_tokens_from_dict, data_packet["grouped_tokens"])
        )
        self.regions = list(map(create_tokens_from_dict, data_packet["regions"]))
        self.tokens_by_block_and_line = {
            block_num: {
                line_num: list(map(create_tokens_from_dict, line_tokens))
                for line_num, line_tokens in block_data.items()
            }
            for block_num, block_data in data_packet["tokens_by_block_and_line"].items()
        }

    def do_OCR(self, verbose: bool = False):
        if not self.tokens:
            ocr_engine = OCREngine()
            self.tokens, self.grouped_tokens, self.tokens_by_block_and_line, self.regions = ocr_engine.OCR(
                self.processed_page, verbose=verbose
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
        OVERLAP_THRESHOLD = 0.3
        max_overlap = 0
        for token in self.grouped_tokens:
            percentage_overlap = token.get_percentage_overlap(
                coordinates, self.page.size
            )
            max_overlap = max(max_overlap, percentage_overlap)
            if percentage_overlap > 0:  # Temporarily setting this to any overlap
                return token
        raise Exception(
            "No significant overlap between token and label at",
            coordinates,
            "was found. Maximum overlap was",
            max_overlap,
        )

    def draw_bounding_boxes(
        self, detail="group", tags=True
    ):  # detail can be group, block, paragraph, line, word
        def draw_rect(canvas: ImageDraw, token: Token, colour: tuple, width: int = 1):
            if tags:  # Display OCR text on top of bounding box
                font = ImageFont.truetype("Andale Mono.ttf")
                canvas.text(
                    (token.coordinates["x"], token.coordinates["y"] - 10),
                    token.text,
                    fill=(0, 0, 0),
                    font=font,
                )

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

        page_copy = self.processed_page.copy().convert("RGB")
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

        if not selected_to_draw:  # If tokens not available, return an empty list
            selected_to_draw = []

        for token in selected_to_draw:
            if token.category:  # Emphasise if this token has been labelled
                draw_rect(canvas, token, (255, 0, 0), 3)
            else:
                draw_rect(canvas, token, (0, 255, 0))

        page_copy.show()


class ObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Token):
            return obj.__dict__
        elif isinstance(obj, InvoicePage):
            return {k: v for k, v in obj.__dict__.items() if k != "page"}
        elif isinstance(obj, Invoice):
            return obj.__dict__
        else:
            return super().default(obj)
