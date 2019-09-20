from typing import List
from Token import Token
import json
import os
from Invoice import Invoice, InvoicePage


class FeatureEngine:
    @classmethod
    def map_labels_to_invoice_OCR(self, data_path: str, verbose: bool = False):
        # This tuple represents the number of pages to do OCR for for each invoice. Eg. (2,1) represents do OCR for the first 2 pages, and for the last page
        RANGE_OF_PAGES_FOR_OCR = (2, 2)
        for filename in os.listdir(data_path):
            if filename.endswith(".pdf"):

                # First check if json tags are present. If they aren't, skip this pdf
                if not os.path.exists(data_path + "/" + filename[:-4] + ".json"):
                    print(
                        "Warning: json tags for",
                        filename,
                        "does not exist. Check if they are in the same folder. Skipping this pdf",
                    )
                    continue

                # Next, do OCR for the relevant pages in the invoice
                invoice = Invoice(data_path + "/" + filename)
                if verbose:
                    print("Processing:", invoice.readable_name)

                if invoice.length() < sum(RANGE_OF_PAGES_FOR_OCR):
                    for page in invoice.pages:
                        page.do_OCR(verbose=verbose)
                else:
                    for page in invoice.pages[: RANGE_OF_PAGES_FOR_OCR[0]]:
                        page.do_OCR(verbose=verbose)
                    for page in invoice.pages[-RANGE_OF_PAGES_FOR_OCR[1] :]:
                        page.do_OCR(verbose=verbose)

                # Try mapping labels
                invoice.map_labels(verbose=verbose)
                invoice.save_data()

    @classmethod
    def create_features(self, token: Token, invoicePage: InvoicePage):
        features = {}

        ### Datatype-related features ###
        # TODO: Implement regexes to determine datatype of this token - DATE, MONEY, NUMBER

        ### Vertical location feature ###
        # TODO: Calculate distance from the top of the image

        ### Date-alignment related features ###
        date_related_tokens = invoicePage.search_tokens("date")

        # Feature 1: Alignment with "date"
        features["aligned_with_date"] = self.create_alignment_feature(
            token, date_related_tokens
        )

        # Feature 2: Min distance to date tokens
        features["distance_to_date_token"] = self.create_min_distance_feature(
            token, date_related_tokens
        )

        ### Invoice number alignment related features ###
        # TODO: Implement features that determine if token is aligned with invoice number related tokens

        ### Tokens nearby ###
        # TODO: Use nearby tokens as feature

        return features

    ####### Functions that help create features go here #######

    @classmethod
    def create_min_distance_feature(self, token: Token, target_tokens: List[Token]):
        """Returns a float which represents the min distance from the token to any of the tokens in the target_tokens list"""
        min_distance = float("inf")
        for target_token in target_tokens:
            distance = token.get_distance_to(target_token)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    @classmethod
    def create_alignment_feature(self, token: Token, target_tokens: List[Token]):
        """Returns a boolean according to whether the token aligns to any of the tokens provided in the target_tokens list"""
        return any(list(map(token.is_aligned_with, target_tokens)))

