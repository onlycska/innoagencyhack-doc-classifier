import os
import pickle

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from trainer.classifier_train import train_classifier
from datetime import datetime
from model.db_tables import tables_dict
import logging


def extract_text_worker(image):
    from preprocessing.text_rotate import img_rotate
    from preprocessing.extraction_test import TextExtractor
    image = image.convert('RGB')
    numpy_image = np.array(image)
    open_cv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    rotated_img = img_rotate(open_cv_image)
    text = TextExtractor().text_extraction_pipeline(rotated_img)
    return text


def classifier_train_worker():
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # выбор данных из таблицы
    from model.db_requests import Database
    from model.db_tables import tables_dict
    table = tables_dict["extracteddata"]
    db_connect = Database()
    data = db_connect.selecting_by_columns(table["name"], columns=table["columns"][:2])
    docs = pd.DataFrame(data, columns=["data", "target"])
    docs.target = le.fit_transform(docs["target"])
    train, test = train_test_split(docs, test_size=0.25, random_state=42, shuffle=True)
    train_data = train["data"].to_numpy()
    train_target = train.target.to_numpy()
    test_data = test["data"].to_numpy()
    test_target = test.target.to_numpy()
    train = {"data": train_data, "target": train_target}
    test = {"data": test_data, "target": test_target}
    first_model, second_model, vocabulary = train_classifier(train, test)
    first_classifier = save_models(first_model, db_connect, vocabulary)
    second_classifier = save_models(second_model, db_connect, vocabulary)
    db_connect.close_conn()
    return first_classifier, second_classifier


def save_models(trained_model, db_connect, model_vocabulary):
    # save first model
    creation_date = datetime.now().isoformat()
    filename = str(creation_date).replace(":", "")

    model_name = rf"D:/doc_processing/classifiers/{filename}.pkl"
    with open(model_name, 'wb') as file:
        pickle.dump(trained_model, file)

    # save model_vocabulary to database
    classifier_table = tables_dict["classifiers"]
    colassifier_models_table = tables_dict["classifiermodels"]
    values = (True, creation_date, 0.8, model_name, 1, creation_date)
    db_connect.inserting(table=classifier_table["name"], values=values, columns=classifier_table["columns"])
    classifier_id = db_connect.selecting_with_condition(classifier_table["name"], "name", model_name)[0][0]
    values = (classifier_id, creation_date, None, 0, model_vocabulary)
    db_connect.inserting(table=colassifier_models_table["name"], values=values,
                         columns=colassifier_models_table["columns"])
    db_connect.commiting()
    return classifier_id


if __name__ == "__main__":
    # print(os.getcwd())
    # raise SystemExit(1)
    classifier_train_worker()
