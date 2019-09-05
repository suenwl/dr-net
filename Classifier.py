import re
from Invoice import InvoicePage
from Token import Token


class Classifier:
    def create_features(self, token: Token, invoicePage: InvoicePage):
        features = {}

        ####### Functions that help create features go here #######

        def create_distance_feature(to_tokens):
            min_distance = float("inf")
            for to_token in to_tokens:
                distance = token.get_distance_to(to_token)
                if distance < min_distance:
                    min_distance = distance
            return min_distance

        def does_alignment_exist(to_tokens):
            return any(list(map(token.is_aligned_with, to_tokens)))

        ### Datatype-related features ###
        # TODO: Implement regexes to determine datatype of the token - DATE, MONEY, NUMBER

        ### Vertical location feature ###
        # TODO: Calculate distance from the top of the image

        ### Date-alignment related features ###
        date_related_tokens = invoicePage.search_tokens("date")

        # Feature 1: Alignment with "date"
        features["aligned_with_date"] = does_alignment_exist(date_related_tokens)

        # Feature 2: Min distance to date tokens
        features["distance_to_date_token"] = create_distance_feature(
            date_related_tokens
        )

        ### Invoice number alignment related features ###
        # TODO: Implement features that determine if token is aligned with invoice number related tokens

        ### Tokens nearby ###
        # TODO: Use nearby tokens as feature

        return features
