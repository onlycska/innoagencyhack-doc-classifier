from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score

from dlnlputils.data import tokenize_text_simple_regex, tokenize_corpus, build_vocabulary, \
    vectorize_texts, SparseFeaturesDataset
from dlnlputils.pipeline import train_eval_loop, predict_with_model, init_random_seed
from typing import Dict

import logging
import matplotlib.pyplot as plt
import torch
import json

from sklearn.datasets import fetch_20newsgroups

import warnings

from postprocessing.json_formatter import NumpyEncoder

warnings.filterwarnings('ignore')


def train_classifier(train_source: Dict, test_source: Dict):
    init_random_seed(42)
    logging.debug(f"Количество обучающих текстов: {len(train_source['data'])}")
    logging.debug(f"Количество тестовых текстов: {len(test_source['data'])}")

    train_tokenized = tokenize_corpus(train_source['data'])
    test_tokenized = tokenize_corpus(test_source['data'])

    print(' '.join(train_tokenized[0]))

    MAX_DF = 0.8
    MIN_COUNT = 4
    vocabulary, word_doc_freq = build_vocabulary(train_tokenized, max_doc_freq=MAX_DF, min_count=MIN_COUNT)
    full_vocabulary_for_db = json.dumps({"model_vocabulary": vocabulary,
                                         "word_doc_freq": word_doc_freq}, cls=NumpyEncoder).encode('utf-8')
    UNIQUE_WORDS_N = len(vocabulary)
    logging.debug(f"Количество уникальных токенов: {UNIQUE_WORDS_N}")
    print(list(vocabulary.items())[:10])

    plt.hist(word_doc_freq, bins=20)
    plt.title('Распределение относительных частот слов')
    plt.yscale('log')

    VECTORIZATION_MODE = 'tfidf'
    train_vectors = vectorize_texts(train_tokenized, vocabulary, word_doc_freq, mode=VECTORIZATION_MODE)
    test_vectors = vectorize_texts(test_tokenized, vocabulary, word_doc_freq, mode=VECTORIZATION_MODE)

    logging.debug(f"Размерность матрицы признаков обучающей выборки: {train_vectors.shape}")
    logging.debug(f"Размерность матрицы признаков обучающей выборки: {test_vectors.shape}")
    logging.debug(f"Количество ненулевых элементов в обучающей выборке: {train_vectors.nnz}")
    logging.debug(f"Процент заполненности матрицы признаков {round(train_vectors.nnz * 100 / (train_vectors.shape[0] * train_vectors.shape[1]), 2)}%")
    logging.debug(f"Количество ненулевых элементов в тестовой выборке: {test_vectors.nnz}")
    logging.debug(f"Процент заполненности матрицы признаков {round(test_vectors.nnz * 100 / (test_vectors.shape[0] * test_vectors.shape[1]), 2)}%")

    # plt.hist(train_vectors.data, bins=20)
    # plt.title('Распределение весов признаков')
    # plt.yscale('log')

    UNIQUE_LABELS_N = len(set(train_source['target']))
    logging.debug(f'Количество уникальных меток: {UNIQUE_LABELS_N}')

    # plt.hist(train_source['target'], bins=np.arange(0, 21))
    # plt.title('Распределение меток в обучающей выборке')
    #
    # plt.hist(test_source['target'], bins=np.arange(0, 21))
    # plt.title('Распределение меток в тестовой выборке')

    train_dataset = SparseFeaturesDataset(train_vectors, train_source['target'])
    test_dataset = SparseFeaturesDataset(test_vectors, test_source['target'])

    model = nn.Linear(UNIQUE_WORDS_N, UNIQUE_LABELS_N)

    scheduler = lambda optim: \
        torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=3, factor=0.5, verbose=True)

    best_val_loss, best_model = train_eval_loop(model=model,
                                                train_dataset=train_dataset,
                                                val_dataset=test_dataset,
                                                criterion=F.cross_entropy,
                                                lr=1e-1,
                                                epoch_n=200,
                                                batch_size=32,
                                                l2_reg_alpha=0,
                                                lr_scheduler_ctor=scheduler)
    first_model = best_model
    train_pred = predict_with_model(best_model, train_dataset)
    train_loss = F.cross_entropy(torch.from_numpy(train_pred),
                                 torch.from_numpy(train_source['target']).long())

    logging.debug(f'Среднее значение функции потерь на обучении: {float(train_loss)}')
    logging.debug(f"Доля верных ответов:{accuracy_score(train_source['target'], train_pred.argmax(-1))}")
    logging.debug(f"F1-мера: {f1_score(train_source['target'], train_pred.argmax(-1), average='macro')}")

    test_pred = predict_with_model(best_model, test_dataset)
    test_loss = F.cross_entropy(torch.from_numpy(test_pred),
                                torch.from_numpy(test_source['target']).long())

    logging.debug(f'Среднее значение функции потерь на обучении: {float(test_loss)}')
    logging.debug(f"Доля верных ответов:{accuracy_score(test_source['target'], test_pred.argmax(-1))}")
    logging.debug(f"F1-мера: {f1_score(test_source['target'], test_pred.argmax(-1), average='macro')}")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    sklearn_pipeline = Pipeline((('vect', TfidfVectorizer(tokenizer=tokenize_text_simple_regex,
                                                          max_df=MAX_DF,
                                                          min_df=MIN_COUNT)),
                                 ('cls', LogisticRegression())))
    sklearn_pipeline.fit(train_source['data'], train_source['target'])
    second_model = sklearn_pipeline

    sklearn_train_pred = sklearn_pipeline.predict_proba(train_source['data'])
    sklearn_train_loss = F.cross_entropy(torch.from_numpy(sklearn_train_pred),
                                         torch.from_numpy(train_source['target']).to(dtype=torch.long))
    logging.debug(f'Среднее значение функции потерь на обучении: {float(sklearn_train_loss)}')
    logging.debug(f"Доля верных ответов:{accuracy_score(train_source['target'], sklearn_train_pred.argmax(-1))}")
    logging.debug(f"F1-мера: {f1_score(train_source['target'], sklearn_train_pred.argmax(-1), average='macro')}")

    sklearn_test_pred = sklearn_pipeline.predict_proba(test_source['data'])
    sklearn_test_loss = F.cross_entropy(torch.from_numpy(sklearn_test_pred),
                                        torch.from_numpy(test_source['target']).to(dtype=torch.long))
    logging.debug(f'Среднее значение функции потерь на обучении: {float(sklearn_test_loss)}')
    logging.debug(f"Доля верных ответов:{accuracy_score(test_source['target'], sklearn_test_pred.argmax(-1))}")
    logging.debug(f"F1-мера: {f1_score(test_source['target'], sklearn_test_pred.argmax(-1), average='macro')}")
    return first_model, second_model, full_vocabulary_for_db
    return second_model, full_vocabulary_for_db

if __name__ == "__main__":
    train = fetch_20newsgroups(subset='train')
    test = fetch_20newsgroups(subset='test')
    print(train.keys())
    print(type(train["data"]))
    print(type(train["data"][0]))
    print(train["target"])
    train_classifier(train, test)
