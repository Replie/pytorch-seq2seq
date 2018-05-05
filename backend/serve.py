#!/usr/bin/python
# coding=utf-8
"""
Author: tal 
Created on 05/05/2018

"""

import os
from backend import predictor
from flask import Flask, jsonify, request, render_template

from seq2seq.util.checkpoint import Checkpoint

app = Flask(__name__)

app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.update(dict(
    DATABASE=os.path.join(app.root_path, 'flaskr.db'),
    SECRET_KEY='development key',
    USERNAME='admin',
    PASSWORD=os.environ.get("DEV_PASS", "SECRET"),
    EXPERIMENT_PATH="/home/project10/project/experiment"
))


def get_checkpoints():
    return os.listdir(os.path.join(app.config.get('EXPERIMENT_PATH'), Checkpoint.CHECKPOINT_DIR_NAME))


checkpoints = get_checkpoints()


def get_args(req):
    if request.method == 'POST':
        args = request.json
    elif request.method == "GET":
        args = request.args
    return args


@app.route("/_predict", methods=["GET", "POST", "OPTIONS"])
def predict():
    args = get_args(request)
    seq_str = args.get('seq_str')
    checkpoint_name = args.get('checkpoint_val')
    suggestions = predictor.predict(app.config.get('EXPERIMENT_PATH'),
                                    checkpoint_name=checkpoint_name,
                                    seq_str=seq_str, n=3)
    return jsonify({"data": {"results": [' '.join(x).strip() for x in suggestions]}})


@app.route("/", methods=["GET"])
def index():
    models = sorted(checkpoints)[-20:]
    return render_template('index.html', models=models)


def main(host="0.0.0.0", port=8080):
    app.run(host=host, port=port, debug=True)


if __name__ == "__main__":
    main()
