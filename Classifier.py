from Invoice import InvoicePage, Invoice
from Token import Token
from FeatureEngine import FeatureEngine
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import random


class Classifier:
    def __init__(self):
        self.models = {
            "Support Vector Machine": None,
            "Neural Network": None,
            "Naive Bayes": None,
            "Random Forest": None,
        }

    def save_model(self, model: str, file_name: str):
        with open(file_name, "wb") as text_file:
            text_file.write(model)

    def train_support_vector_machine(self, data, labels):
        parameters = {
            "kernel": ("linear", "rbf"),
            "gamma": [0.001, 0.0001],
            "C": [1, 100],
        }
        # original line without use of grid search to optimise parameters
        # classifier = svm.SVC(gamma=0.001, C=100.0)
        svc = svm.SVC()
        classifier = GridSearchCV(svc, parameters, cv=5)
        classifier.fit(data, labels)
        self.models["Support Vector Machine"] = classifier
        # save the model to disk
        self.save_model(pickle.dump(classifier), "svm_model")

    def train_neural_network(self, data, labels):
        # multi-layer perceptron (MLP) algorithm
        # consider increasing neuron number to match number of features as data set
        classifier = MLPClassifier(
            solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
        )
        classifier.fit(data, labels)
        self.models["Neural Network"] = classifier
        self.save_model(pickle.dumps(classifier), "nn_model")

    def train_naive_bayes(self, data, labels):
        classifier = GaussianNB()
        classifier.fit(data, labels)
        self.models["Naive Bayes"] = classifier
        self.save_model(pickle.dumps(classifier), "nb_model")

    def train_random_forest(self, data, labels):
        classifier = RandomForestClassifier(
            n_estimators=100, max_depth=2, random_state=0
        )
        classifier.fit(data, labels)
        self.models["Random Forest"] = classifier
        self.save_model(pickle.dumps(classifier), "rf_model")

    @classmethod
    def create_train_and_test_packet(
        cls, pathname: str, percentage_train: float = 0.8, verbose: bool = False
    ):
        invoices = FeatureEngine.map_labels_to_invoice_OCR(pathname, verbose)
        random.shuffle(invoices)
        splitting_point = int(len(invoices) * 0.8)
        return {"train": invoices[:splitting_point], "test": invoices[splitting_point:]}

    def train(self, model_name: str, data, labels):
        # mlp sensitive to feature scaling, plus NN requires this so we standardise scaling first
        scaler = StandardScaler()
        scaler.fit(data)
        data = scaler.transform(data)
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        # labels = scaler.transform(labels)
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
            model_file = model_name + ".sav"
            if os.path.exists(model_file):
                loaded_model = pickle.load(open(model_file, "rb"))
                result = loaded_model.predict(input_features)
            else:
                raise Exception(
                    "The model either has not been trained, or has not been loaded correctly. Call the train() method, and check if the model is in the correct directory"
                )
        else:
            model = self.models[model_name]
            return model.predict(input_features)
