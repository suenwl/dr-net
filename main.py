#%%
# import methods defined in the other files
from OCREngine import OCREngine
from FeatureEngine import FeatureEngine
from Invoice import Invoice
from Token import Token
from Classifier import Classifier
from util import features_to_use

from flask import Flask
from flask import render_template
from flask_socketio import SocketIO, send, emit

import time
import json

app = Flask(__name__, static_folder = 'build/static', template_folder='build')
app.config['SECRET_KEY'] = 'mysecret'
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on('req-home')
def handle_home():
    data = {'test':321}
    jsn = json.dumps(data)
    emit('res-home', jsn)


if __name__ == '__main__':
    socketio.run(app, debug=True)


'''

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
