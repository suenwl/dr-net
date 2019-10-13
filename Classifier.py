from Invoice import InvoicePage, Invoice
from Token import Token
from FeatureEngine import FeatureEngine
from util import features_to_use, category_mappings
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.metrics import classification_report
import pickle
import os
import random
import numpy as np

# testing new feature selection
# import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


class Classifier:
    def __init__(self):
        self.models = {
            "Support Vector Machine": None,
            "Neural Network": None,
            "Naive Bayes": None,
            "Random Forest": None,
        }

    def save(self):
        with open("classifier", "wb") as text_file:
            text_file.write(pickle.dumps(self))

    def load(self):
        if os.path.exists("classifier"):
            classifier = pickle.load(open("classifier", "rb"))
            self.models = classifier.models
        else:
            raise Exception("Classifier save file does not exist")

    def train_support_vector_machine(self, data, labels):
        parameters = {
            "kernel": ("linear", "rbf"),
            "gamma": [0.001, 0.0001],
            "C": [1, 100],
        }
        # original line without use of grid search to optimise parameters
        # classifier = svm.SVC(gamma=0.001, C=100.0)
        classifier = svm.SVC(probability=True, gamma=0.001, C=1, kernel="linear")
        # classifier = GridSearchCV(classifier, parameters, cv=5)
        classifier.fit(data, labels)
        self.models["Support Vector Machine"] = classifier
        # save the model to disk
        # self.save()

    def train_neural_network(self, data, labels):
        # multi-layer perceptron (MLP) algorithm
        # consider increasing neuron number to match number of features as data set
        classifier = MLPClassifier(
            solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(20, 20), random_state=1
        )
        classifier.fit(data, labels)
        self.models["Neural Network"] = classifier
        # self.save()

    def train_naive_bayes(self, data, labels):
        classifier = GaussianNB()
        classifier.fit(data, labels)
        self.models["Naive Bayes"] = classifier
        # self.save()

    def train_random_forest(self, data, labels):
        classifier = RandomForestClassifier(
            n_estimators=100, max_depth=2, random_state=0
        )
        classifier.fit(data, labels)
        self.models["Random Forest"] = classifier
        # self.save()

    def train(self, model_name: str, data, labels, max_features="all"):
        # mlp sensitive to feature scaling, plus NN requires this so we standardise scaling first
        data = normalize(data)
        labels = list(map(lambda label: category_mappings[label], labels))
        # self.feature_selector = SelectKBest(chi2, k=max_features).fit(data, labels)
        # data = self.feature_selector.transform(data)
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
        self.save()

    @classmethod
    def get_data_and_labels(
        cls, invoice_list, features_to_use=features_to_use, scale_others=False
    ):
        number_of_non_others_tokens = 0
        number_of_others_tokens = 0
        OTHERS_SCALING_FACTOR = 0.3  # Maximum percentage of others tokens

        def get_feature_list(token, invoice_page):
            features = FeatureEngine.create_features(token, invoice_page)
            return [v for k, v in features.items() if k in features_to_use]

        data = []
        labels = []
        tokens = []
        for invoice in invoice_list:
            for invoice_page in invoice.pages:
                if invoice_page.grouped_tokens:
                    for token in invoice_page.grouped_tokens:
                        if not scale_others:
                            data.append(get_feature_list(token, invoice_page))
                            labels.append(token.category)
                            tokens.append(token)
                        else:
                            if token.category != "Others":
                                number_of_non_others_tokens += 1
                                data.append(get_feature_list(token, invoice_page))
                                labels.append(token.category)
                                tokens.append(token)
                            elif (
                                number_of_others_tokens
                                < number_of_non_others_tokens * OTHERS_SCALING_FACTOR
                            ):
                                number_of_others_tokens += 1
                                data.append(get_feature_list(token, invoice_page))
                                labels.append(token.category)
                                tokens.append(token)

        return data, labels, tokens

    @classmethod
    def create_train_and_test_packet(
        cls, invoices, features_to_use, percentage_train: float = 0.8
    ):
        random.shuffle(invoices)
        splitting_point = int(len(invoices) * 0.8)
        train_invoices = invoices[:splitting_point]
        test_invoices = invoices[splitting_point:]

        train_data, train_labels, train_tokens = cls.get_data_and_labels(
            train_invoices, features_to_use, scale_others=False
        )
        test_data, test_labels, train_tokens = cls.get_data_and_labels(
            test_invoices, features_to_use, scale_others=False
        )

        return {
            "train_data": train_data,
            "train_labels": train_labels,
            "test_data": test_data,
            "test_labels": test_labels,
        }

    def predict_token_classifications(self, input_features, model_name: str):
        """ Predicts the classification of a token using a model """
        if not self.models[model_name]:
            raise Exception(
                "Model is either not trained, or not loaded. Call the load() method if you have a classifier save file"
            )
        model = self.models[model_name]

        # input_features = self.feature_selector.transform(input_features)
        predictions = model.predict(input_features)
        prediction_probabilities = model.predict_proba(input_features)
        prediction_confidence = [
            prediction_probabilities[i][category]
            for i, category in enumerate(predictions)
        ]
        return {"categories": predictions, "confidence": prediction_confidence}

    def predict_invoice_fields(self, invoice: Invoice, model_name: str):
        predicted_categories = {
            "Account number": (None, 0),
            "Consumption period": (None, 0),
            "Country of consumption": (None, 0),
            "Currency of invoice": (None, 0),
            "Date of invoice": (None, 0),
            "Invoice number": (None, 0),
            "Name of provider": (None, 0),
            "Others": (None, 0),
            "PO Number": (None, 0),
            "Tax": (None, 0),
            "Total amount": (None, 0),
        }
        data, labels, tokens = self.get_data_and_labels(
            [invoice], features_to_use, scale_others=False
        )
        predictions = self.predict_token_classifications(data, model_name)
        categories = list(
            map(lambda label: category_mappings[label], predictions["categories"])
        )
        confidence = predictions["confidence"]
        for index in range(len(categories)):
            category = categories[index]
            if confidence[index] > predicted_categories[category][1]:
                predicted_categories[category] = (tokens[index], confidence[index])

        return predicted_categories

    def prediction_summary(self, predictions, labels):
        text_predictions = list(
            map(lambda label: category_mappings[label], predictions["categories"])
        )
        report = "'" + classification_report(labels, text_predictions)[1:]
        print(report)

    @classmethod
    def select_features(cls, data, labels, max_number="all"):
        select = SelectKBest(chi2, k=max_number).fit(data, labels)
        list(select.scores_)

    @classmethod
    # bugs to fix to resolve: rfe = rfe.fit(train_data,train_labels)
    # to do after: pick num of features that maximises accuracy
    def recursive_feature_elimination(
        self, model_name: str, train_data, train_labels, test_data, test_labels
    ):
        model = self.models["Support Vector Machine"]
        train_data = normalize(train_data)
        train_labels = LabelEncoder().fit_transform(train_labels)
        num_features = 20
        # high_score=0
        # Variable to store the optimum features
        # nof=0
        # score_list =[]
        rfe = RFE(model, num_features)
        rfe = rfe.fit(train_data, train_labels)
        # print summaries for the selection of attributes
        print(rfe.support_)
        print(rfe.ranking_)

