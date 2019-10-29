#%%
# import methods defined in the other files
from OCREngine import OCREngine
from FeatureEngine import FeatureEngine
from Invoice import Invoice
from Token import Token
from Classifier import Classifier
from config import features_to_use

from flask import Flask
from flask import render_template
from flask_socketio import SocketIO, send, emit

import time
import json

app = Flask(__name__, static_folder="build/static", template_folder="build")
app.config["SECRET_KEY"] = "mysecret"
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("req-home")
def handle_home():
    data = {"test": 321}
    jsn = json.dumps(data)
    emit("res-home", jsn)


if __name__ == "__main__":
    socketio.run(app, debug=True)
