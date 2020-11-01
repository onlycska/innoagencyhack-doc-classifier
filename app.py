from flask import Flask
from flask import request

from os import getcwd, mkdir, path, walk
from datetime import datetime
from preprocessing.extraction_test import TextExtractor
from predictors.doc_type_predictor import classify
import urllib.parse

import flask
# TODO переделать на flask.logging
import logging
import json

app = Flask(__name__)

logs_folder_path = getcwd() + r"\logs"
if not path.exists(logs_folder_path):
    mkdir(logs_folder_path)
log_path = rf"{logs_folder_path}\{datetime.date(datetime.now())}.log"
filemode = "a" if path.exists(log_path) else "w"
logging.basicConfig(format=u'%(levelname)-8s [%(asctime)s] %(message)s',
                    level=logging.DEBUG,
                    filename=log_path,
                    filemode=filemode,
                    datefmt='%d-%b-%y %H:%M:%S')


@app.route('/api/textextractor/extract')
def extract_text_to_db():
    try:
        logging.debug(f"Request for data extracting has received.")
        filepath = request.headers.get("path")
        filepath = urllib.parse.unquote(filepath)
        json = TextExtractor.doc_worker(filepath)
        if json is None:
            flask.abort(500)
        return json
    except Exception as e:
        logging.exception(e)


@app.route('/api/textextractor/extract-many')
def extract_many_texts_to_db():
    req_path = request.headers.get("path")
    req_path = urllib.parse.unquote(req_path)
    # если передана папка, все документы из неё забираются и из её подпапок.
    # типами документов становятся имена конечных папок из директории, в которой они лежат
    if request.headers.get("isFolder"):
        folder = []
        files_to_extract = {}
        for i in walk(req_path):
            folder.append(i)
        for address, dirs, files in folder:
            for filepath in files:
                files_to_extract[address + "\\" + filepath] = address.rsplit("\\", 1)[1]
    # иначе должен быть передан словарь вида "путь к изображению": "тип документа"
    else:
        files_to_extract = req_path
        assert files_to_extract, dict
    print(files_to_extract)
    for filepath, doc_type in files_to_extract.items():
        logging.info(f"Extraction text started from file: {filepath}")
        TextExtractor.doc_worker(filepath, doc_type)
    return app.make_response(("OK", 201))


@app.route("/api/classifier/train-from-db")
def train_classifier_from_db():
    from preprocessing.workers import classifier_train_worker
    classifiers = classifier_train_worker()
    return app.make_response((f"model created. classifiers id: {classifiers}", 201))


@app.route("/api/classifier/classify-by-path")
def classify_doc():
    doc_type_dict = {
        0: "БТИ",
        1: "ЗУ",
        2: "Разр. на ввод",
        3: "Разр. на стр-во",
        4: "Свид. АГР"
    }

    filepath = request.headers.get("path")
    filepath = urllib.parse.unquote(filepath)
    classifier_id = int(request.headers.get("classifier_id"))
    json_ans = TextExtractor.doc_worker(filepath)
    json_ans = json.loads(json_ans)
    type = classify(json_ans["full_text"], classifier_id)[0]
    type = doc_type_dict[type]
    return app.make_response((f"document type: {type}", 200))


if __name__ == '__main__':
    app.run()
