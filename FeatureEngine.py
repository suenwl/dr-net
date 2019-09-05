from typing import List
from Token import Token
from Invoice import InvoicePage


class FeatureEngine:
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

    # Returns a float which represents the min distance from the token to any of the tokens in the target_tokens list
    def create_min_distance_feature(self, token: Token, target_tokens: List[Token]):
        min_distance = float("inf")
        for target_token in target_tokens:
            distance = token.get_distance_to(target_token)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    # Returns a boolean according to whether the token aligns to any of the tokens provided in the target_tokens list
    def create_alignment_feature(self, token: Token, target_tokens: List[Token]):
        return any(list(map(token.is_aligned_with, target_token)))
