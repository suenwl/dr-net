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
import os

from flask import Flask
from flask import render_template
from flask import request, session
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from threading import Thread, Event
from time import sleep
import enum
from werkzeug.utils import secure_filename

import time
import json
import csv
import yaml

import mock_data


app = Flask(__name__, static_folder="build/static", template_folder="build")
app.config["SECRET_KEY"] = "mysecret"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///invoices.db"
socketio = SocketIO(app, cors_allowed_origins="*")
db = SQLAlchemy(app)

with open(r"config.yaml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

invoice_dir = config["invoice_dir"]
invoice_categories = config["categories"]
clean_database = config["clean_database"]

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
    category = db.Column(db.String)

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
    def __init__(self, db):
        self.db = db

    def emit_invoices_update(self):
        invoices = self.db.get_all_invoices_w_data()
        for i in invoices:
            if "_sa_instance_state" in i:
                del i["_sa_instance_state"]
        jsn = json.dumps(invoices)
        socketio.emit("invoices_update", jsn)

    def emit_status_update(self, msg):
        jsn = json.dumps(msg)
        socketio.emit("status_update", jsn)

    def emit_metrics(self, metrics):
        jsn = json.dumps(metrics)
        socketio.emit("accuracy_metrics", jsn)

    def emit_num_invoices_update(self):
        invoices = self.db.get_all_invoices_w_data()
        for i in invoices:
            if "_sa_instance_state" in i:
                del i["_sa_instance_state"]
        jsn = json.dumps(invoices)
        socketio.emit("num_invoices_update", jsn)


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
        try:
            with open("invoice_db.csv", "w") as output_file:
                dict_writer = csv.DictWriter(output_file, fields)
                dict_writer.writeheader()
                dict_writer.writerows(data_to_write)
        except Exception as e:
            print("Cannot write to file: ", e)

    def insert_invoice(self, data):
        i = self.model(**data)
        self.db.session.add(i)
        self.db.session.commit()
        self.update_csv()

    def update_invoice(self, id, data):
        for field in data:
            col = field["name"].lower().replace(" ", "_")
            val = field["value"]
            inv = self.db.session.query(self.model).get(id)
            setattr(inv, col, val)
        setattr(inv, "status", "reviewed")
        self.db.session.commit()

    def update_status(self, id, new_status):
        self.model.query.get(id).status = new_status
        self.db.session.commit()
        self.update_csv()

    def update_results(self, id, predictions):
        for key in predictions:
            formatted_key = key.lower().replace(" ", "_")
            inv = self.db.session.query(self.model).get(id)
            val = predictions[key][0]
            if val:
                setattr(inv, formatted_key, str(val))
                if key == "Name of provider":
                    setattr(inv, "category", "Others")
                    for sub_cat in invoice_categories:
                        if val in invoice_categories[sub_cat]:
                            setattr(inv, "category", sub_cat)
                try:
                    setattr(inv, formatted_key + "_conf", float(predictions[key][1]))
                except:
                    setattr(inv, formatted_key + "_conf", -0.0001)
            else:
                setattr(inv, formatted_key, "No Prediction")
                setattr(inv, formatted_key + "_conf", 0.0)
            self.update_status(id, "processed")
        self.update_csv()


invoice_database = InvoiceDataBase(db)
socket_emitter = Emmitter(invoice_database)
classifier = Classifier()
classifier.load()
if clean_database:
    invoice_database.destroy()
    invoice_database.create()


class WatcherThread(Thread):
    def __init__(self):
        self.delay = 1
        self.process_queue = []
        self.mock_queue = []
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
                    print(f"ERROR: {file} not processed: ", e)
                finally:
                    socket_emitter.emit_invoices_update()

            print("all done")
            self.process_queue = []

    def run(self):

        # # start inserting mock data ===============================
        # for df in mock_data.dataFile:
        #     inv_id = str(random()) + ".pdf"
        #     inv_id = inv_id.replace("0.", "invoice_id_")
        #     self.invoice_db.insert_invoice({"id": inv_id})
        #     self.invoice_db.update_results(inv_id, df)
        #     socket_emitter.emit_invoices_update()
        # # end inserting mock data ===============================

        while not thread_stop_event.isSet():
            self.get_new_files()
            self.process_new_files()
            # print("starting new watch cycle")
            sleep(self.delay)


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("req_invoices")
def handle_invoices_req():
    socket_emitter.emit_invoices_update()


@socketio.on("req_metrics")
def handle_metrics_req():
    metrics = classifier.model_metrics["Neural Network"]
    socket_emitter.emit_metrics(metrics)


@socketio.on("req_invoice_details_update")
def handle_update_invoice_details(data):
    invoice_database.update_invoice(data["filename"], data["data"])
    socket_emitter.emit_status_update(
        {
            "title": "INVOICE UPDATED",
            "content": f"{data['filename']} has been reviewed and updated",
        }
    )
    socket_emitter.emit_invoices_update()


@socketio.on("connect")
def test_connect():
    print(f"Client id:{request.sid} connected")
    run_watcher()


@socketio.on("disconnect")
def test_disconnect():
    print("Client disconnected")

UPLOAD_FOLDER = invoice_dir
ALLOWED_EXTENSIONS = set(['pdf', 'png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def fileUpload():
    print('sdfsdfd')
    target=UPLOAD_FOLDER
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file'] 
    filename = secure_filename(file.filename)
    destination="/".join([target, filename])
    file.save(destination)
    session['uploadFilePath']=destination
    response="File Uploaded"
    return response


def run_watcher():
    global thread
    if not thread.isAlive():
        print(" ============ Watcher Thread Started ============ ")
        thread = WatcherThread()
        thread.start()


if __name__ == "__main__":
    socketio.run(app)

