"""
This script loads data from a directory, and trains the classifier.
The classifier is then saved as a pickle, ready for use in prediction
"""

#%%
from OCREngine import OCREngine
from FeatureEngine import FeatureEngine
from Invoice import Invoice
from Token import Token
from Classifier import Classifier
from config import features_to_use
from util import missing_fields_percentage


TRAINING_DATA_DIR = "/Users/suenwailun/Sync Documents/University/Y4S1/BT3101 Business Analytics Capstone Project/Training data"


print("Starting...")
invoices = FeatureEngine.load_invoices_and_map_labels(
    TRAINING_DATA_DIR, autoload=False, verbose=True
)
#%%
print("\nCreating training and testing data...")
data = Classifier.create_train_and_test_packet(invoices, features_to_use)

#%%
classifier = Classifier()
print("Training classifier with features of dimension", len(data["train_data"][0]))
classifier.train("Neural Network", data["train_data"], data["train_labels"])
predictions = classifier.predict_token_classifications(
    data["test_data"], "Neural Network"
)
classifier.prediction_summary(
    predictions=predictions, labels=data["test_labels"], model="Neural Network"
)

#%%
import json

invoices_perf = classifier.sort_invoices_by_predictive_accuracy(
    invoices, "Neural Network"
)
with open("invoice scores.json", "w") as f:
    f.write(json.dumps(invoices_perf))
print("Worst 20 performers:")
for invoice in invoices_perf[:20]:
    print(
        f"Name of invoice: {invoice['name']}     Accuracy: {invoice['overall_accuracy']}"
    )

#%%
# Calculate missing field percentages
classifier = Classifier()
classifier.load()
predictions_before_cleaning = []
predictions = []
invoice_names = [invoice.readable_name for invoice in invoices]
for invoice in invoices:
    pred = classifier.predict_invoice_fields(invoice, "Neural Network")
    finalised_pred = classifier.finalise_output(pred, invoice)
    predictions_before_cleaning.append(pred)
    predictions.append(finalised_pred)
print()
print(
    "Missing fields percentages after padding", missing_fields_percentage(predictions)
)

# Write output to csv
classifier.write_predictions_to_csv(predictions, invoice_names)


# %%

