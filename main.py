#%%
# import methods defined in the other files
from OCREngine import OCREngine
from FeatureEngine import FeatureEngine
from Invoice import Invoice
from Token import Token
from Classifier import Classifier
from util import *


print("Starting...")
# Load invoices in specific folder
INVOICE_PATH = "/Users/theia/Documents/Data/Year 4 Sem 1/BT3101 BUSINESS ANALYTICS CAPSTONE/Invoices/singtel_2.pdf"
invoice = Invoice(INVOICE_PATH)
page = invoice.get_page(1)
# page.remove_lines()
page.do_OCR()
#page.draw_bounding_boxes("group")
feature_engine = FeatureEngine()
classifier = Classifier()

#%% Demo 1: create features for each token on the page
for token in page.grouped_tokens:
    features = feature_engine.create_features(token, page)
    print(token)
    print(len(features), "features")
    print(features)
    print(" ")
"""
#%% Demo 2: Print tokens grouped by blocks
for i, block in page.get_tokens_by_block().items():
    print(block)
    print(" ")
"""
#%% Demo 3: Print page
# page.page.resize((600, 900))
# page.draw_bounding_boxes("word")
# print(list(map(lambda x :x.date_values,page.grouped_tokens)))
# print(list(map(lambda x :x.get_currency(),page.grouped_tokens)))
# print(list(map(lambda x :x.get_num_label(),page.grouped_tokens)))
# print(list(map(lambda x :x.get_total_label(),page.grouped_tokens)))
# print(list(map(lambda x :x.contains_digit,page.grouped_tokens)))
# print(list(map(lambda x :[x.coordinates, x.text],page.grouped_tokens)))
