print("Starting...")
invoices = FeatureEngine.load_invoices_and_map_labels(
    "C:/Users/theia/Documents/Data/Year 4 Sem 1/BT3101 BUSINESS ANALYTICS CAPSTONE/Invoices",
    autoload=False,
    verbose=True,
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
classifier.prediction_summary(predictions=predictions, labels=data["test_labels"])

#%%
invoice = Invoice(
    "C:/Users/theia/Documents/Data/Year 4 Sem 1/BT3101 BUSINESS ANALYTICS CAPSTONE/Sales Invoice_test.pdf"
)
with open("invoice scores.json", "w") as f:
    f.write(json.dumps(invoices_perf))
print("Worst 20 performers:")
for invoice in invoices_perf[:20]:
    print(
        f"Name of invoice: {invoice['name']}     Accuracy: {invoice['overall_accuracy']}"
    )

#%%
'''
