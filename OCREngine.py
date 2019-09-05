# Text processing methods go here
from invoice2data import extract_data
from invoice2data.extract.loader import read_templates

import pytesseract
import pandas as pd
from pandas import DataFrame
from PIL.PpmImagePlugin import PpmImageFile as Image

from Token import Token


class OCREngine:
    def clean_OCR_output(self, raw_OCR_output: DataFrame):

        UNECESSARY_CHARACTERS = [" ", "(", ")", "&", ";"]

        without_null = raw_OCR_output.loc[raw_OCR_output["text"].notnull()]

        without_unecessary_characters = without_null.loc[
            ~without_null["text"].isin(UNECESSARY_CHARACTERS)
        ]  # Not in unecessary characters
        return without_unecessary_characters

    def convert_ocr_dataframe_to_token_list(self, ocr_dataframe: DataFrame):
        token_list = []

        for index, row in ocr_dataframe.iterrows():

            token_list.append(
                Token(
                    row.text,
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

    def OCR(self, image: Image):
        import time

        start_time = time.time()
        raw_OCR_output = pytesseract.image_to_data(image, output_type="data.frame")
        print("--- %s seconds ---" % (time.time() - start_time))
        cleaned_OCR_output = self.clean_OCR_output(raw_OCR_output)
        tokens = self.convert_ocr_dataframe_to_token_list(cleaned_OCR_output)

        return tokens
