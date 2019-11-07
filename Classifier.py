from Invoice import InvoicePage, Invoice
from Token import Token
from FeatureEngine import FeatureEngine
from util import is_hori_aligned,is_vert_aligned
from config import features_to_use, category_mappings
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
import pickle
import os
import random
import numpy as np
import csv
import re

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
        self.model_metrics = {
            "Support Vector Machine": None,
            "Neural Network": None,
            "Naive Bayes": None,
            "Random Forest": None,
        }

    def save(self):
        with open("classifier.pkl", "wb") as text_file:
            text_file.write(pickle.dumps(self))

    def load(self):
        if os.path.exists("classifier.pkl"):
            classifier = pickle.load(open("classifier.pkl", "rb"))
            self.model_metrics = classifier.model_metrics
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
        classifier = svm.SVC(probability=True, gamma="scale", C=1, kernel="linear")
        # classifier = GridSearchCV(classifier, parameters, cv=5)
        classifier.fit(data, labels)
        self.models["Support Vector Machine"] = classifier
        # save the model to disk

    def train_neural_network(self, data, labels):
        # multi-layer perceptron (MLP) algorithm
        # consider increasing neuron number to match number of features as data set
        classifier = MLPClassifier(
            solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(30, 30, 30), random_state=1
        )
        classifier.fit(data, labels)
        self.models["Neural Network"] = classifier

    def train_naive_bayes(self, data, labels):
        classifier = GaussianNB()
        classifier.fit(data, labels)
        self.models["Naive Bayes"] = classifier

    def train_random_forest(self, data, labels):
        classifier = RandomForestClassifier(
            n_estimators=100, max_depth=2, random_state=0
        )
        classifier.fit(data, labels)
        self.models["Random Forest"] = classifier

    def train(self, model_name: str, data, labels, max_features="all"):
        # mlp sensitive to feature scaling, plus NN requires this so we standardise scaling first
        labels = list(map(lambda label: category_mappings[label], labels))
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
        # random.shuffle(invoices)
        # splitting_point = int(len(invoices) * 0.8)
        # train_invoices = invoices[:splitting_point]
        # test_invoices = invoices[splitting_point:]

        # train_data, train_labels, train_tokens = cls.get_data_and_labels(
        #     train_invoices, features_to_use, scale_others=False
        # )
        # test_data, test_labels, train_tokens = cls.get_data_and_labels(
        #     test_invoices, features_to_use, scale_others=False
        # )

        data, labels, tokens = cls.get_data_and_labels(
            invoices, features_to_use, scale_others=False
        )

        # Compile data into a dictionary
        zipped = list(zip(data, labels, tokens))
        dictionary_of_categories = {}
        for token in zipped:
            label = token[1]
            if label not in dictionary_of_categories:  # If category does not yet exist
                dictionary_of_categories[label] = []
            else:
                dictionary_of_categories[label].append(token)

        train = []
        test = []

        # Shuffle all categories
        for category in dictionary_of_categories:
            category_data = dictionary_of_categories[category]
            random.shuffle(category_data)
            splitting_point = int(len(category_data) * percentage_train)
            train.extend(category_data[:splitting_point])
            test.extend(category_data[splitting_point:])

        random.shuffle(train)
        random.shuffle(test)

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        for data in train:
            train_data.append(data[0])
            train_labels.append(data[1])
        for data in test:
            test_data.append(data[0])
            test_labels.append(data[1])

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
        map_category_to_position = lambda category: list(model.classes_).index(category)
        predictions = model.predict(input_features)
        prediction_probabilities = model.predict_proba(input_features)
        prediction_confidence = [
            prediction_probabilities[i][map_category_to_position(category)]
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
                predicted_categories[category] = [tokens[index], confidence[index]]

        return predicted_categories

    @classmethod
    def clean_output(cls,predicted_categories,invoice):
        predicted_categories.pop("Others",None)
        for key in predicted_categories:
            relevant_token = predicted_categories[key][0]
            if not relevant_token:
                continue

            if key == "Account number" or key == "Invoice number":
                pattern = re.compile('[\W_]+', re.UNICODE) # By Python definition '\W == [^a-zA-Z0-9_], which excludes all numbers, letters and _
                predicted_categories[key][0] = re.sub(pattern,"",relevant_token.text)
            elif key == "Consumption period":
                if relevant_token.date_range:
                    start_date = relevant_token.date_range[0][0]
                    end_date = relevant_token.date_range[1][0]
                    predicted_categories[key][0] = ";".join([start_date,end_date])
                else:
                    predicted_categories[key] = (None,0)
            elif key == "Country of consumption":
                if any(country in relevant_token.text.lower() for country in ["singapore", "sg"]):
                    predicted_categories[key][0] = "Singapore"
                elif any(country in relevant_token.text.lower() for country in ["hong kong","hk"]):
                    predicted_categories[key][0] = "Hong Kong"
                elif any(country in relevant_token.text.lower() for country in ["japan","jp"]):
                    predicted_categories[key][0] = "Japan"
            elif key == "Currency of invoice":
                if any(currency in relevant_token.text.lower() for currency in ["sgd", "sg$", "$sg", "s$","singapore dollar"]):
                    predicted_categories[key][0] = "SGD"
                elif any(currency in relevant_token.text.lower() for currency in ["hkd", "hk$", "$hk", "hong kong dollar"]):
                    predicted_categories[key][0] = "HKD"
                elif any(currency in relevant_token.text.lower() for currency in ["jpy", "¥", "yen"]):
                    predicted_categories[key][0] = "JPY"
            elif key == "Date of invoice":
                if relevant_token.date_values:
                    predicted_categories[key][0] = relevant_token.date_values[0]
                else:
                    predicted_categories[key] = (None,0)
            elif key == "Tax" or key == "Total amount":
                output = re.search("[\d,]+[.]{0,1}[\d]{0,4}",relevant_token.text)
                if output:
                    predicted_categories[key][0] = output.group(0)
            else:
                predicted_categories[key][0] = relevant_token.text

        return RulesBasedClassifier.pad_missing_predictions(predicted_categories,invoice)


    def sort_invoices_by_predictive_accuracy(self, invoices, model_name: str):
        """
        Takes in a list of invoices, gets their respective predictions using a particular model,
        and then returns a dictionary for each invoice with the overall accuracy, and boolean
        for each category to signify whether that category was predicted correctly
        :returns : A list of dictionaries with three keys (name, overall_accuracy, and detailed_accuracy),
            sorted in an ascending manner according to the overall accuracy
        """
        invoice_predictive_accuracies = []

        for invoice in invoices:
            predicted_categories = self.predict_invoice_fields(invoice, model_name)
            predictive_categories_bool = {
                k: bool(v[0] and k == v[0].category)
                for (k, v) in predicted_categories.items()
            }
            predictive_accuracy = {
                "name": invoice.readable_name,
                "overall_accuracy": sum(predictive_categories_bool.values())
                / len(predictive_categories_bool),
                "detailed_accuracy": predictive_categories_bool,
            }
            invoice_predictive_accuracies.append(predictive_accuracy)
        return sorted(
            invoice_predictive_accuracies,
            key=lambda predictive_accuracy: predictive_accuracy["overall_accuracy"],
        )

    def prediction_summary(self, predictions, labels, model):
        text_predictions = list(
            map(lambda label: category_mappings[label], predictions["categories"])
        )
        report = "'" + classification_report(labels, text_predictions)[1:]
        print(report)
        self.model_metrics[model] = classification_report(labels, text_predictions,output_dict=True)
        self.save()

    @classmethod
    def write_predictions_to_csv(cls,predictions,invoice_names):
        """
        Writes predictions, which is a list of dictionaries, to a csv file
        """
        keys = ["Invoice"] + list(predictions[0].keys())
        # Add invoice name to data
        for index, prediction in enumerate(predictions):
            for field in prediction:
                prediction[field] = prediction[field][0]
            prediction["Invoice"] = invoice_names[index]
        
        with open('invoice_predictions.csv', 'w') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(predictions)


    @classmethod
    def select_features(cls, data, labels, max_number="all"):
        select = SelectKBest(chi2, k=max_number).fit(data, labels)
        list(select.scores_)

class RulesBasedClassifier:
    @classmethod
    def pad_missing_predictions(cls,predictions,invoice):

        # predicted_categories = {
        #     "Account number": (None, 0),
        #     "Consumption period": (None, 0),
        #     "Country of consumption": (None, 0),
        #     "Currency of invoice": (None, 0),
        #     "Date of invoice": (None, 0),
        #     "Invoice number": (None, 0),
        #     "Name of provider": (None, 0),
        #     "Others": (None, 0),
        #     "PO Number": (None, 0),
        #     "Tax": (None, 0),
        #     "Total amount": (None, 0),
        # }

        # Obtain predictions which are missing
        missing_predictions = [key for (key,value) in predictions.items() if value[0] == None]
        missing_predictions.remove("PO Number")

        for field in missing_predictions:
            
            if field == "Country of consumption":
                predictions["Country of consumption"] = [cls.get_country(invoice),"Rules based"]

            elif field == "Currency of invoice":
                if predictions["Country of consumption"][0]:
                    predictions["Currency of invoice"] = [cls.get_currency(predictions),"Rules based"]
                else:
                    predictions["Country of consumption"] = [cls.get_country(invoice),"Rules based"]
                    predictions["Currency of invoice"] = [cls.get_currency(predictions),"Rules based"]
            
        return predictions

    
    @classmethod
    def nearest_hori_aligned_token(cls,pivot_token,page):
        smallest_dist = None
        best_token = None
        
        for token in page.grouped_tokens:
            if is_hori_aligned(pivot_token,token,10):
                dist = calc_min_dist(pivot_token,token)
                if dist < smallest_dist:
                    smallest_dist = dist
                    best_token = token

        return best_token

    @classmethod
    def get_country(cls,invoice):
        country_lookup = {
            "singapore" : "Singapore",
            "japan": "Japan", 
            "hk": "Hong Kong",
            "hong kong": "Hong Kong"
        }

        country_counts = {
                    "Singapore": 0,
                    "Hong Kong": 0,
                    "Japan": 0
                }
        for page in invoice.pages:
            for token in page.grouped_tokens:
                for country in country_lookup:
                    if country in token.text:
                        detected_country = country_lookup[country]
                        country_counts[detected_country] += 1
        
        replacement_country = max(country_counts.items(),key=lambda country: country[1])[0]
        return replacement_country

    @classmethod
    def get_currency(cls,predictions):
        currency_lookup = {
            "Singapore": "SGD",
            "Hong Kong": "HKD",
            "Japan": "JPY"
        }
        country = predictions["Country of consumption"][0]
        return currency_lookup[country]