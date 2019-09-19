from Invoice import InvoicePage, Invoice
from Token import Token
from sklearn import svm
import pickle
import os
import json


class Classifier:
    def __init__(self):
        self.models = {
            "Support Vector Machine": None,
            "Neural Network": None,
            "Naive Bayes": None,
            "Random Forest": None,
        }

    def create_training_data(self, training_data_path: str, verbose=False):
        # This tuple represents the number of pages to do OCR for for each invoice. Eg. (2,1) represents do OCR for the first 2 pages, and for the last page
        NUMBER_OF_PAGES_FOR_OCR = (2, 2)
        for filename in os.listdir(training_data_path):
            if filename.endswith(".pdf"):

                # First check if json tags are present. If they aren't, skip this pdf
                try:
                    json_tags = json.load(
                        open(training_data_path + "/" + filename[:-4] + ".json", "r")
                    )
                except IOError:
                    print(
                        "Warning: json tags for",
                        filename,
                        "does not exist. Check if they are in the same folder. Skipping this pdf",
                    )
                    continue

                # Next, do OCR for the relevant pages in the invoice
                invoice = Invoice(training_data_path + "/" + filename)
                if verbose:
                    print("Processing:", invoice.readable_name)

                if invoice.length() < sum(NUMBER_OF_PAGES_FOR_OCR):
                    for page in invoice.pages:
                        page.do_OCR(verbose=verbose)
                else:
                    for page in invoice.pages[: NUMBER_OF_PAGES_FOR_OCR[0]]:
                        page.do_OCR(verbose=verbose)
                    for page in invoice.pages[-NUMBER_OF_PAGES_FOR_OCR[1] :]:
                        page.do_OCR(verbose=verbose)

                # Try mapping labels
                invoice.map_labels(verbose=verbose)

    def save_model(self, model: str, file_name: str):
        with open(file_name, "w") as text_file:
            text_file.write(model)

    def train_support_vector_machine(self, data, labels):
        classifier = svm.SVC(gamma=0.001, C=100.0)
        classifier.fit(data, labels)
        self.models["Support Vector Machine"] = classifier
        self.save_model(pickle.dumps(classifier), "svm_model")

    def train_neural_network(self, data, labels):
        pass

    def train_naive_bayes(self, data, labels):
        pass

    def train_random_forest(self, data, labels):
        pass

    def train(self, model_name: str, data, labels):
        """ Used to train a specific model """
        if model_name == "Support Vector Machine":
            self.train_support_vector_machine(data, labels)
        elif model_name == "Neural Network":
            self.train_neural_network(data, labels)
        elif model_name == "Naive Bayes":
            self.train_naive_bayes(data, labels)
        elif model_name == "Random Forest":
            self.train_random_forest(data, labels)

    def predict(self, input_features, model_name: str):
        """ Predicts the classification of a token using a model """
        if not self.models["model"]:
            raise Exception(
                "The model either has not been trained, or has not been loaded correctly. Call the train() method, and check if the model is in the correct directory"
            )
        else:
            model = self.models[model_name]
            return model.predict(input_features)
