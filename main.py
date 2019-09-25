#%%
# import methods defined in the other files
from OCREngine import OCREngine
from FeatureEngine import FeatureEngine
from Invoice import Invoice
from Token import Token
from Classifier import Classifier
from util import *

print("Starting...")
data = Classifier.create_train_and_test_packet(
    "/Users/suenwailun/Sync Documents/University/Y4S1/BT3101 Business Analytics Capstone Project/circles invoices",
    verbose=True,
)

classifier = Classifier()
classifier.train("Support Vector Machine", data["train_data"], data["train_labels"])
classifier.predict(data["test_data"], "Support Vector Machine")

#%% Demo 2: Print tokens grouped by blocks
"""
for i, block in page.tokens_by_block_and_line.items():
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
