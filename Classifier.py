from Invoice import InvoicePage, Invoice
from Token import Token
from FeatureEngine import FeatureEngine
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
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
        self.label_encoder = LabelEncoder()

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
        svc = svm.SVC(probability=True, verbose=True)
        classifier = GridSearchCV(svc, parameters, cv=5)
        classifier.fit(data, labels)
        self.models["Support Vector Machine"] = classifier
        # save the model to disk
        self.save_model(pickle.dumps(classifier), "svm_model")

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
    def get_data_and_labels(cls, invoice_list):
        number_of_non_others_tokens = 0
        number_of_others_tokens = 0
        OTHERS_SCALING_FACTOR = 0.3  # Maximum percentage of others tokens

        feature_selection_strings = ["vert_align", "hori_align", "contains", "rel_dist"]

        def get_feature_list(token, invoice_page, feature_selection_strings):
            features = FeatureEngine.create_features(token, invoice_page)
            all_features = []
            for selection_string in feature_selection_strings:
                sel_features = [v for k, v in features.items() if selection_string in k]
                all_features.extend(sel_features)
            print(all_features)
            return all_features

        data = []
        labels = []
        for invoice in invoice_list:
            for invoice_page in invoice.pages:
                if invoice_page.grouped_tokens:
                    for token in invoice_page.grouped_tokens:
                        if token.category != "Others":
                            number_of_non_others_tokens += 1
                            data.append(
                                get_feature_list(
                                    token, invoice_page, feature_selection_strings
                                )
                            )
                            labels.append(token.category)
                        elif (
                            number_of_others_tokens
                            < number_of_non_others_tokens * OTHERS_SCALING_FACTOR
                        ):
                            number_of_others_tokens += 1
                            data.append(
                                get_feature_list(
                                    token, invoice_page, feature_selection_strings
                                )
                            )
                            labels.append(token.category)

        return data, labels

    @classmethod
    def create_train_and_test_packet(
        cls, pathname: str, percentage_train: float = 0.8, verbose: bool = False
    ):
        invoices = FeatureEngine.map_labels_to_invoice_OCR(pathname, True, verbose)
        random.shuffle(invoices)
        splitting_point = int(len(invoices) * 0.8)
        train_invoices = invoices[:splitting_point]
        test_invoices = invoices[splitting_point:]

        train_data, train_labels = cls.get_data_and_labels(train_invoices)
        test_data, test_labels = cls.get_data_and_labels(test_invoices)

        return {
            "train_data": train_data,
            "train_labels": train_labels,
            "test_data": test_data,
            "test_labels": test_labels,
        }

    def train(self, model_name: str, data, labels):
        # mlp sensitive to feature scaling, plus NN requires this so we standardise scaling first
        data = normalize(data)
        labels = self.label_encoder.fit_transform(labels)
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
        if not self.models[model_name]:
            if os.path.exists(model_name):
                model = pickle.load(open(model_name, "rb"))
            else:
                raise Exception(
                    "The model either has not been trained, or has not been loaded correctly. Call the train() method, and check if the model is in the correct directory"
                )
        else:
            model = self.models[model_name]

        return model.predict(input_features)

    def prediction_summary(self, predictions, labels):
        fmt = "{:<8}{:<30}{}"
        num_correct = 0
        correct_counts_per_category = {k: 0 for k in labels}
        category_instances = {k: 0 for k in labels}

        print(fmt.format(" ", "Prediction", "Actual"))
        for i, (prediction, label) in enumerate(zip(predictions, labels)):
            print(fmt.format(i, prediction, label))
            if prediction == label:
                num_correct += 1
                correct_counts_per_category[label] = (
                    correct_counts_per_category[label] + 1
                )
            category_instances[label] = category_instances[label] + 1

        print("")
        print("==========================================")
        print("Overall Accuracy:", str(num_correct * 100 / len(predictions)) + "%")
        print("==========================================")
        print(fmt.format("", "Category", "Accuracy"))
        for category in category_instances:
            print(
                fmt.format(
                    "",
                    category,
                    correct_counts_per_category[category]
                    * 100
                    / category_instances[category],
                )
            )

