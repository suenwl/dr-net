# Text processing methods go here
# from invoice2data import extract_data
# from invoice2data.extract.loader import read_templates

import pytesseract
import pandas as pd
import nltk

# nltk.download("stopwords") #Use if nltk stopwords not downloaded
from nltk.corpus import stopwords

from pandas import DataFrame
from PIL import Image

from Token import Token

# for windows since brew does not work, is there a better way of doing this?
# pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
class OCREngine:
    def clean_OCR_output(self, raw_OCR_output: DataFrame):
        """ Cleans the OCR output by removing uncessary characters """

        UNECESSARY_CHARACTERS = [" ", "(", ")", "&", ";", "|"]

        without_null = raw_OCR_output.loc[raw_OCR_output["text"].notnull()]

        without_unecessary_characters = without_null.loc[
            ~without_null["text"].isin(UNECESSARY_CHARACTERS)
        ]  # Not in unecessary characters
        return without_unecessary_characters

    def convert_ocr_dataframe_to_token_list(self, ocr_dataframe: DataFrame):
        """ Use the OCR results, a dataframe, to generate a list of tokens """
        token_list = []

        for index, row in ocr_dataframe.iterrows():

            token_list.append(
                Token(
                    row.text if type(row.text) == str else None,
                    {
                        "x": row.left,
                        "y": row.top,
                        "height": row.height,
                        "width": row.width,
                    },
                    row.conf,
                    {
                        "page_num": row.page_num,
                        "block_num": row.block_num,
                        "par_num": row.par_num,
                        "line_num": row.line_num,
                        "word_num": row.word_num,
                    },
                )
            )

        return token_list

    def get_regions(self, raw_OCR_output: DataFrame):
        """ Gets regions of text based on the OCR output """
        return raw_OCR_output.loc[raw_OCR_output["text"].isnull()]

    def get_tokens_by_block_and_lines(self, tokens):
        """ Gets tokens by block and lines, returning a dictionary nested first by token blocks and then by lines"""

        blocks_and_lines = {}

        for token in tokens:
            block_num = token.token_structure["block_num"]
            line_num = token.token_structure["line_num"]
            if block_num in blocks_and_lines:
                if line_num in blocks_and_lines[block_num]:
                    blocks_and_lines[block_num][line_num].append(token)
                else:
                    blocks_and_lines[block_num][line_num] = [token]
            else:
                blocks_and_lines[block_num] = {line_num: [token]}

        return blocks_and_lines

    def group_tokens(self, blocks_and_lines):
        """ Group tokens together based on their proximity to other tokens, creating a more meaningful token list """

        def horizontal_distance_between(token1, token2):
            if token1.coordinates["x"] > token2.coordinates["x"]:
                return (
                    token1.coordinates["x"]
                    - token2.coordinates["x"]
                    - token2.coordinates["width"]
                )
            else:
                return (
                    token2.coordinates["x"]
                    - token1.coordinates["x"]
                    - token1.coordinates["width"]
                )

        def combine_tokens_into_one_token(token_list):
            x = y = float("inf")
            bottom_right_x = bottom_right_y = 0
            text = ""

            for token in token_list:
                if not text:
                    text = token.text
                else:
                    text = text + " " + token.text

                x = min(token.coordinates["x"], x)
                y = min(token.coordinates["y"], y)
                bottom_right_x = max(
                    token.coordinates["x"] + token.coordinates["width"], bottom_right_x
                )
                bottom_right_y = max(
                    token.coordinates["y"] + token.coordinates["height"], bottom_right_y
                )

            return Token(
                text,
                {
                    "x": x,
                    "y": y,
                    "width": bottom_right_x - x,
                    "height": bottom_right_y - y,
                },
                "NA",
                {**token.token_structure, "word_num": "Grouped"},
            )

        grouped_tokens = []

        for block in blocks_and_lines:
            for line in blocks_and_lines[block]:
                current_line = blocks_and_lines[block][line]
                current_group = []
                ADJUSTMENT_FACTOR = 10

                for token in current_line:
                    if current_group:
                        height_of_current_group = max(
                            list(
                                map(
                                    lambda token: token.coordinates["height"],
                                    current_group,
                                )
                            )
                        )
                        TOO_FAR = (
                            horizontal_distance_between(token, current_group[-1])
                            > height_of_current_group / 2 + ADJUSTMENT_FACTOR
                        )

                        LAST_TOKEN_ENDS_WITH_COLON = current_group[-1].text[-1] == ":"
                        ALIGNED_HORIZONTALLY = token.is_horizontally_aligned_with(
                            current_group[-1]
                        )

                    if not current_group:
                        current_group.append(token)
                    elif (
                        TOO_FAR
                        or LAST_TOKEN_ENDS_WITH_COLON
                        or not ALIGNED_HORIZONTALLY
                    ):  # This token should not be combined into the current group
                        grouped_tokens.append(
                            combine_tokens_into_one_token(current_group)
                        )
                        current_group = [token]  # Reset the current group
                    else:  # This token is close enough to add to the current group
                        current_group.append(token)

                if current_group:
                    grouped_tokens.append(combine_tokens_into_one_token(current_group))

        return grouped_tokens

    def remove_stopwords(self, tokens):
        stopwords_set = set(stopwords.words("english"))
        return list(filter(lambda t: t.text not in stopwords_set, tokens))

    def OCR(self, image: Image, verbose:bool=False):
        import time

        start_time = time.time()
        # Note for pytesseract output:
        # level 1: page; level 2: block; level 3: paragraph; level 4: line; level 5: word
        # raw_OCR_output = pytesseract.image_to_data(image, output_type="data.frame")
        raw_OCR_output = pytesseract.image_to_data(
            image,
            output_type="data.frame",
            config="pitsync_linear_version==6, textord_noise_rejwords==0， textord_noise_rejrows==0",
        )
        if verbose:
            print("--- Processed page in %s seconds ---" % (time.time() - start_time))

        # Do some preliminary processing and grouping of the raw OCR output
        cleaned_OCR_output = self.clean_OCR_output(raw_OCR_output)
        tokens = self.convert_ocr_dataframe_to_token_list(cleaned_OCR_output)
        tokens_by_blocks_and_lines = self.get_tokens_by_block_and_lines(tokens)

        grouped_tokens = self.group_tokens(tokens_by_blocks_and_lines)
        tokens_without_stopwords = self.remove_stopwords(tokens)

        regions = self.convert_ocr_dataframe_to_token_list(
            self.get_regions(raw_OCR_output)
        )

        return (tokens, grouped_tokens, tokens_by_blocks_and_lines, regions)
