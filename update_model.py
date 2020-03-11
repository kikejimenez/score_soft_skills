import os, logging
import gensim
import pandas as pd
import numpy as np

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)
YES_FILE = 'training_data/yes_sents.csv'
NO_FILE = 'training_data/no_sents.csv'
MODEL_FILE = 'gensim_model'
TYPES_FILE = 'list_soft_skills.txt'  # www.resumegenius.com


def read_input(input_file):
    """This method reads the input file"""
    for line in open(input_file):
        yield remove_stopwords(line)


def make_doc_df(file):
    documents = pd.Series(read_input(file)).drop_duplicates()
    tokens = documents.apply(lambda doc: simple_preprocess(doc))
    tokens = tokens[tokens.apply(lambda x: len(x) > 0).values]
    df = pd.DataFrame([documents, tokens]).T
    df.columns = ['document', 'token']
    logging.info("Done reading data file")
    return df.dropna()


def main():

    soft_skills = make_doc_df(TYPES_FILE)
    sentences = pd.concat([make_doc_df(YES_FILE), make_doc_df(NO_FILE)])
    corpus = sentences.token.append(soft_skills.token).copy()
    model = gensim.models.Word2Vec(
        corpus,
        size=100 + 4 * 8,
        window=7,
        min_count=1,
        workers=10,
        batch_words=900
    )
    model.train(
        corpus, total_examples=corpus.shape[0], epochs=20, compute_loss=True
    )

    model.save(MODEL_FILE)


if __name__ == "__main__":
    main()