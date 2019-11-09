#%%
# import methods defined in the other files
from OCREngine import OCREngine
from FeatureEngine import FeatureEngine
from Invoice import Invoice
from Token import Token
from Classifier import Classifier
from config import features_to_use
from random import random
from os import listdir

from flask import Flask
from flask import render_template
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, send, emit
from threading import Thread, Event
from time import sleep
import enum

import time
import json
import csv

app = Flask(__name__, static_folder="build/static", template_folder="build")
app.config["SECRET_KEY"] = "mysecret"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///invoices.db"
socketio = SocketIO(app, cors_allowed_origins="*")
db = SQLAlchemy(app)

invoice_dir = "/Users/suenwailun/Sync Documents/University/Y4S1/BT3101 Business Analytics Capstone Project/Training data 2"

thread = Thread()
thread_stop_event = Event()


class InvoiceData(db.Model):
    id = db.Column(db.String, primary_key=True)
    account_number = db.Column(db.String)
    consumption_period = db.Column(db.String)
    country_of_consumption = db.Column(db.String)
    currency_of_invoice = db.Column(db.String)
    date_of_invoice = db.Column(db.String)
    invoice_number = db.Column(db.String)
    name_of_provider = db.Column(db.String)
    po_number = db.Column(db.String)
    tax = db.Column(db.String)
    total_amount = db.Column(db.String)

    account_number_conf = db.Column(db.Float)
    consumption_period_conf = db.Column(db.Float)
    country_of_consumption_conf = db.Column(db.Float)
    currency_of_invoice_conf = db.Column(db.Float)
    date_of_invoice_conf = db.Column(db.Float)
    invoice_number_conf = db.Column(db.Float)
    name_of_provider_conf = db.Column(db.Float)
    po_number_conf = db.Column(db.Float)
    tax_conf = db.Column(db.Float)
    total_amount_conf = db.Column(db.Float)

    status = db.Column(db.String, nullable=False, default="unprocessed")

    def __repr__(self):
        return self.id


class Emmitter:
    def __init__(self, skt, db):
        self.skt = skt
        self.db = db

    def emit_invoices_update(self):
        invoices = self.db.get_all_invoices_w_data()
        for i in invoices:
            if "_sa_instance_state" in i:
                del i["_sa_instance_state"]
        jsn = json.dumps(invoices)
        self.skt.emit("invoices_update", jsn)

    def emit_status_update(self, msg):
        jsn = json.dumps(msg)
        socketio.emit("status_update", jsn)

    def emit_metrics(self, metrics):
        jsn = json.dumps(metrics)
        socketio.emit("accuracy_metrics", jsn)


class InvoiceDataBase:
    def __init__(self, db):
        self.db = db
        self.model = InvoiceData

    def create(self):
        return self.db.create_all()

    def destroy(self):
        return self.db.drop_all()

    def get_all_invoices(self):
        return self.model.query.all()

    def get_invoice_data(self, inv_name):
        return self.model.query.get(inv_name).__dict__

    def get_all_invoices_w_data(self):
        return [invoice.__dict__ for invoice in self.get_all_invoices()]

    def contains(self, filename):
        return (
            self.db.session.query(self.model.id).filter_by(id=filename).scalar()
            is not None
        )

    def update_csv(self):
        data = self.get_all_invoices_w_data()
        fields = [
            "id",
            "name_of_provider",
            "po_number",
            "tax",
            "account_number",
            "total_amount",
            "consumption_period",
            "country_of_consumption",
            "currency_of_invoice",
            "date_of_invoice",
            "invoice_number",
        ]
        data_to_write = [
            {k: v for k, v in invoice.items() if k in fields} for invoice in data
        ]
        with open("invoice_db.csv", "w") as output_file:
            dict_writer = csv.DictWriter(output_file, fields)
            dict_writer.writeheader()
            dict_writer.writerows(data_to_write)

    def insert_invoice(self, data):
        i = self.model(**data)
        self.db.session.add(i)
        self.db.session.commit()
        self.update_csv()

    def update_status(self, id, new_status):
        self.model.query.get(id).status = new_status
        self.db.session.commit()
        self.update_csv()

    def update_results(self, id, predictions):
        for key in predictions:
            formatted_key = key.lower().replace(" ", "_")
            inv = self.db.session.query(InvoiceData).get(id)
            val = predictions[key][0]
            if val:
                setattr(inv, formatted_key, str(val))
                setattr(inv, formatted_key + "_conf", float(predictions[key][1]))
            else:
                setattr(inv, formatted_key, "No Prediction")
                setattr(inv, formatted_key + "_conf", float(predictions[key][1]))
            self.update_status(id, "processed")
        self.update_csv()


invoice_database = InvoiceDataBase(db)
socket_emitter = Emmitter(socketio, invoice_database)
classifier = Classifier()
classifier.load()
invoice_database.destroy()
invoice_database.create()


class WatcherThread(Thread):
    def __init__(self):
        self.delay = 1
        self.process_queue = []
        self.invoice_db = invoice_database
        self.classifier = classifier
        super(WatcherThread, self).__init__()

    @staticmethod
    def get_file_extension(filename):
        return filename.split(".")[-1]

    def get_new_files(self):
        """
        Scans for list of all files in directory and add PDF files which are not processed yet into processing queue
        """
        all_files = listdir(invoice_dir)
        pdf_only = filter(
            lambda f: WatcherThread.get_file_extension(f) == "pdf", all_files
        )
        for pdf_file in pdf_only:
            # get from database, see if it is there
            if (
                not self.invoice_db.contains(pdf_file)
                or self.invoice_db.get_invoice_data(pdf_file)["status"] == "unprocessed"
            ):
                # Insert unprocessed row in database
                self.invoice_db.insert_invoice({"id": pdf_file})
                self.process_queue.append(pdf_file)

    def process_file(self, file):
        print("Processing " + file)

        # Create Invoice objects
        invoice = Invoice(invoice_dir + "/" + file)
        invoice.do_OCR()

        # Get predictions
        predictions = self.classifier.finalise_output(
            self.classifier.predict_invoice_fields(invoice, "Neural Network"), invoice
        )
        return predictions

    def process_new_files(self):
        num_new_files = len(self.process_queue)
        if num_new_files > 0:
            print(f"found {num_new_files} files")
            socket_emitter.emit_status_update(
                {
                    "title": f"FOUND {num_new_files} NEW FILE(S)",
                    "content": " | ".join(self.process_queue),
                }
            )
            socket_emitter.emit_invoices_update()
            for file in self.process_queue:
                self.invoice_db.update_status(file, "processing")
                socket_emitter.emit_status_update(
                    {"title": "PROCESSING", "content": file}
                )
                socket_emitter.emit_invoices_update()
                try:
                    print(f"processing {file}")
                    predictions = self.process_file(file)
                    self.invoice_db.update_results(file, predictions)
                    socket_emitter.emit_status_update(
                        {"title": "PROCESSING COMPLETED", "content": file}
                    )
                    print(f"{file} processing completed")
                except Exception as e:
                    self.invoice_db.update_status(file, "processing failed")
                    socket_emitter.emit_status_update(
                        {"title": "PROCESSING ERROR", "content": file}
                    )
                    print(f"ERROR: {id} not processed", e)
                finally:
                    socket_emitter.emit_invoices_update()

            print("all done")
            self.process_queue = []

    def run(self):
        while not thread_stop_event.isSet():
            self.get_new_files()
            self.process_new_files()
            print("starting new watch cycle")
            sleep(self.delay)


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("req_invoices")
def handle_invoices_req():
    socket_emitter.emit_invoices_update()


@socketio.on("req_metrics")
def handle_metrics_req():
    metrics = classifier.model_metrics
    print(metrics)
    socket_emitter.emit_metrics(metrics)


@socketio.on("connect")
def test_connect():
    # need visibility of the global thread object
    global thread
    print("Client connected")

    # Start the random number generator thread only if the thread has not been started before.
    if not thread.isAlive():
        print("Starting Thread")
        thread = WatcherThread()
        thread.start()


@socketio.on("disconnect")
def test_disconnect():
    print("Client disconnected")


if __name__ == "__main__":
    socketio.run(app)

# if __name__ == "__main__":
# conn = sqlite3.connect('test.db')
# socketio.run(app, debug=True)


"""
    def test(self):
        d = {
            "id": str(random()) + ".pdf",
            "account_number": "1234567",
            "consumption_period": "01012010",
            "country_of_consumption": "hk",
            "currency_of_invoice": "hkd",
            "date_of_invoice": "01012010",
            "invoice_number": "in202130-1",
            "name_of_provider": "west",
            "po_number": "po32ds-32",
            "tax": 0.0,
            "total_amount": 3000.0,
            "account_number_conf": random(),
            "consumption_period_conf": random(),
            "country_of_consumption_conf": random(),
            "currency_of_invoice_conf": random(),
            "date_of_invoice_conf": random(),
            "invoice_number_conf": random(),
            "name_of_provider_conf": random(),
            "po_number_conf": random(),
            "tax_conf": random(),
            "total_amount_conf": random(),
            "status": "unprocessed",
        }
        self.insert_invoices(d)
"""

