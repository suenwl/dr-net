# import methods defined in the other files
from OCREngine import OCREngine
from FeatureEngine import FeatureEngine
from Invoice import Invoice
from Token import Token
from Classifier import Classifier
from util import *

if __name__ == "__main__":
    # Load invoices in specific folder
    invoice = Invoice(
        "/Users/suenwailun/Sync Documents/University/Y4S1/BT3101 Business Analytics Capstone Project/Tech demo/starhub.pdf"
    )
    page = invoice.get_page(1)
    page.do_OCR()
    feature_engine = FeatureEngine()

    # Test 1: create features for each token on the page
    for token in page.tokens:
        print(token, feature_engine.create_features(token, page))

    # Test 2: Print tokens grouped by blocks
    # for i, block in page.get_tokens_by_block().items():
    #     print(block)
    #     print(" ")
