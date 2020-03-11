import os, sys, logging
from pathlib import Path
import gensim
import pandas as pd
import numpy as np

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

MODEL_FILE = 'gensim_model'
TYPES_FILE = 'list_soft_skills.txt'  # www.resumegenius.com

MODEL = gensim.models.Word2Vec.load(MODEL_FILE)


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


def process_sentence(sentence):
    return [
        word for word in simple_preprocess(remove_stopwords(sentence))
        if word in MODEL.wv.vocab
    ]


def get_score_pd(sentences, soft_skills):
    '''Get Cosine similarity of a list of sentences'''
    docs = [process_sentence(sent) for sent in sentences]
    flatten_docs = [d for doc in docs for d in doc]
    soft_skills['similarities'] = soft_skills.token.apply(
        lambda sk: MODEL.wv.n_similarity(sk, flatten_docs)
    )
    result = soft_skills[[
        'document', 'similarities'
    ]].sort_values(by=['similarities'], ascending=True)[-5:]
    result.reset_index(drop=True, inplace=True)
    result.columns = ['soft_skills', 'score']
    result.index = pd.Index([1, 2, 3, 4, 5], name='top')
    return result[['soft_skills']]


def main(sentences_file):

    sentences = [line for line in open(sentences_file)]
    soft_skills = make_doc_df(TYPES_FILE)
    print(get_score_pd(sentences, soft_skills))


if __name__ == "__main__":
    sentences_file = sys.argv[1]
    main(sentences_file)
