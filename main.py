#%%
# import methods defined in the other files
from OCREngine import OCREngine
from FeatureEngine import FeatureEngine
from Invoice import Invoice
from Token import Token
from Classifier import Classifier
from util import features_to_use

print("Starting...")
print("Loading invoices...")
invoices = FeatureEngine.load_invoices_and_map_labels(
    "/Users/suenwailun/Sync Documents/University/Y4S1/BT3101 Business Analytics Capstone Project/Training data",
    autoload=True,
    verbose=True,
)
data = Classifier.create_train_and_test_packet(invoices, features_to_use)

classifier = Classifier()
print("Training classifier...")
classifier.train("Support Vector Machine", data["train_data"], data["train_labels"])
predictions = classifier.predict(data["test_data"], "Support Vector Machine")
classifier.prediction_summary(
    classifier.label_encoder.inverse_transform(predictions), data["test_labels"]
)


#%%
