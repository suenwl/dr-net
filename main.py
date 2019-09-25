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
INVOICE_PATH = "/Users/ng-ka/OneDrive/Desktop/BT3101/starhub_7.pdf"
        for root, directories, files in os.walk("/home/user/PDFs", topdown=True):
        invoice = Invoice(INVOICE_PATH)
        for i in range(1,len(invoice.pages)+1):
            page = invoice.get_page(i)
        # page.remove_lines()
            page.do_OCR()
#page.draw_bounding_boxes("group")
feature_engine = FeatureEngine()
classifier = Classifier()

json_labels = "/Users/ng-ka/OneDrive/Desktop/BT3101/starhub_7.json"
invoice.map_labels(json_labels, False)
#%% Demo 1: create features for each token on the page
training_data = []
label_data = []
training_reference = []
for i in range(1,len(invoice.pages)+1):
    for token in invoice.get_page(i).grouped_tokens:
        features = feature_engine.create_features(token, page)
        #print(token)
    #print(len(features), "features")
    #print(features)
    #print(" ")
        features_list = list(features.values())
        training_data.append(features_list)
        label_data.append(token.category) 
        training_reference.append(features)

for i in range(len(label_data)):
    #if(type(label_data[i]) == NoneType)
    if label_data[i] is None: # check for nonetype class, presents problems in label encoding later
        label_data[i] = "Others"
    #lst.append(type(label_data[i]))
    #lst2.append(label_data[i])

#print(train_len)
#print(train_ref_len)
#print(label_len)
#print(training_data)
#print(label_data)

classifier.train("Support Vector Machine", training_data, label_data)
#classifier.predict(input_features, "Support Vector Machine")
"""
#%% Demo 2: Print tokens grouped by blocks
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
