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
INVOICE_PATH = "/Users/suenwailun/Sync Documents/University/Y4S1/BT3101 Business Analytics Capstone Project/Tech demo/starhub.pdf"
invoice = Invoice(INVOICE_PATH)
page = invoice.get_page(1)
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

#%%
