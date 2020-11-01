import pickle
import json
import numpy as np
from sklearn.pipeline import Pipeline

from model.db_requests import Database
from model.db_tables import tables_dict
from dlnlputils.data import tokenize_text_simple_regex, tokenize_corpus, build_vocabulary, \
    vectorize_texts, SparseFeaturesDataset
from dlnlputils.pipeline import predict_with_model, init_random_seed


def classify(doc_text, classifier_id):
    db_connect = Database()
    doc_text_list = [doc_text, ]
    # загружаем нужный классификатор
    classifiers_table = tables_dict["classifiers"]
    full_path = db_connect.selecting_with_condition(classifiers_table["name"], "classifierid", classifier_id)[0][4]
    with open(full_path, 'rb') as file:
        ai_model = pickle.load(file)
    if type(ai_model) is Pipeline:
        test_pred = ai_model.predict_proba(doc_text_list)
        return test_pred.argmax(-1)
    # загружаем словарь токенов классификатор
    classifiers_models_table = tables_dict["classifiermodels"]
    full_vocabulary = db_connect.selecting_with_condition(classifiers_models_table["name"],
                                                          "classifierid", classifier_id)[0][5]
    db_connect.close_conn()
    full_vocabulary = bytes(full_vocabulary)
    vocabulary_json = json.loads(full_vocabulary)
    vocabulary = vocabulary_json["model_vocabulary"]
    word_doc_freq = np.array(vocabulary_json["word_doc_freq"])

    VECTORIZATION_MODE = 'tfidf'
    tokenized_text = tokenize_corpus(doc_text_list)
    vectors = vectorize_texts(tokenized_text, vocabulary, word_doc_freq, mode=VECTORIZATION_MODE)
    predicted_data = SparseFeaturesDataset(vectors, np.array([0]))

    test_pred = predict_with_model(ai_model, predicted_data)
    return test_pred.argmax(-1)


if __name__ == "__main__":
    text = """ПРАВИТЕЛЬСТВО МОСКВЫ сы МОСКОВСКИЙ ЗЕМЕЛЬНЫЙ КОМИТЕТ ДОГОВОР О ПРЕДОСТАВЛЕНИИ УЧАСТКА в пользование на условиях аренды (договор аренды земли) №_ М-02-141516 Москомзем Архив ДОГОВОР АРЕНДЫ ЗЕМЕЛЬНОГО УЧАСТКА (краткосрочной) № М-02-141516 МОСКВА "И ширляе 2002г. — Московский земельный комитет (Москомзем), именуемый в дальнейшем «Арендодатель», в лице начальника Объединения регулирования | землепользования Москомзема в Северо-Восточном административном округе г.Москвы Мареняна Игоря Борисовича, действующего на основании Положения и доверенности от 09.01.2002г. № 33-И-342/1-(2), от имени Мэрии (Администрации) г.Москвы, с одной Стороны, и Открытое акционерное общество "САНДУН МКТС»”, именуемое в дальнейшем "Арендатор”, в лице генерального директора Паскаль Моретти, действующего на основании Устава, с другой Стороны"""
    test = classify(text, 9)
    print(test)
