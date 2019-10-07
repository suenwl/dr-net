#%%
# import methods defined in the other files
from OCREngine import OCREngine
from FeatureEngine import FeatureEngine
from Invoice import Invoice
from Token import Token
from Classifier import Classifier
from util import features_to_use

print("Starting...")
invoices = FeatureEngine.load_invoices_and_map_labels(
    "/Users/suenwailun/Sync Documents/University/Y4S1/BT3101 Business Analytics Capstone Project/Training data",
    autoload=True,
    verbose=True,
)
#%%
print("\nCreating training and testing data...")
data = Classifier.create_train_and_test_packet(invoices, features_to_use)
classifier = Classifier()
print("Training classifier...")
classifier.train("Neural Network", data["train_data"], data["train_labels"])
predictions = classifier.predict_token_classifications(
    data["test_data"], "Neural Network"
)
classifier.prediction_summary(predictions=predictions, labels=data["test_labels"])
# classifier.recursive_feature_elimination("Support Vector Machine", data["train_data"], data["train_labels"], data["test_data"], data["test_labels"])


#%%
invoice = Invoice(
    "/Users/suenwailun/Sync Documents/University/Y4S1/BT3101 Business Analytics Capstone Project/Training data/circles_1.pdf"
)
invoice.do_OCR(verbose=True)
classifier.predict_invoice_fields(invoice, "Neural Network")

#%%
