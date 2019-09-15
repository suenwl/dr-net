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
#INVOICE_PATH = "/Users/candicetay/Documents/Academic/NUS/Year 4/BT3101 Capstone NatWest/PDF Invoices/Singtel Aug.pdf"
#INVOICE_PATH = "/Users/candicetay/Documents/Academic/NUS/Year 4/BT3101 Capstone NatWest/PDF Invoices/Pest Control Service Contract.pdf"
#INVOICE_PATH = "/Users/candicetay/Documents/Academic/NUS/Year 4/BT3101 Capstone NatWest/PDF Invoices/Dive Receipt.pdf"
#INVOICE_PATH = "/Users/candicetay/Documents/Academic/NUS/Year 4/BT3101 Capstone NatWest/PDF Invoices/Circles April 18.pdf"
INVOICE_PATH = "/Users/theia/Documents/Data/Year 4 Sem 1/BT3101 BUSINESS ANALYTICS CAPSTONE/Invoices/Circles April 18.pdf"
invoice = Invoice(INVOICE_PATH)
page = invoice.get_page(1)
# page.remove_lines()
page.do_OCR()
feature_engine = FeatureEngine()

#%% Demo 1: create features for each token on the page
for token in page.tokens:
    print(token, feature_engine.create_features(token, page))
    print(" ")

#%% Demo 2: Print tokens grouped by blocks
for i, block in page.get_tokens_by_block().items():
    print(block)
    print(" ")

#%% Demo 3: Print page
page.page.resize((600, 900))
page.draw_bounding_boxes("word")
#%%
#%%
