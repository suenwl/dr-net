import re
from Invoice import InvoicePage
from Token import Token


class Classifier:
    def create_features(self, token: Token, invoicePage: InvoicePage):
        features = {}

        ### Datatype-related features ###

        ### Date-related features ###
        date_related_tokens = invoicePage.search_tokens("date")

        # Feature 1: Alignment with "date"
        lst = any(list(map(token.is_aligned_with, date_related_tokens)))
        if lst:
            features["aligned_with_date"] = True
        else:
            features["aligned_with_date"] = False

        # Feature 2: Min distance to date tokens
        min_distance = float("inf")
        for date_token in date_related_tokens:
            distance = token.get_distance_to(date_token)
            if distance < min_distance:
                min_distance = distance
        features["distance_to_date_token"] = min_distance

        #

        return features
