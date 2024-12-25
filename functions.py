import matplotlib.pyplot as plt
import gensim
import spacy
import nltk
import pandas as pd
from gensim.models import LsiModel, TfidfModel
from gensim.models.coherencemodel import CoherenceModel

def tfidf_corpus(doc_term_metrix):
    tfidf = TfidfModel(corpus = doc_term_metrix, normalize = True)
    corpus_tfidf = tfidf[doc_term_metrix]
    return corpus_tfidf


def get_coherence_scores(corpus, dictionary, text, min_topics, max_topics):
    coherence_values = []
    model_list = []
    for num_topics_i in range(min_topics, max_topics+1):
        model = LsiModel(corpus, num_topics = num_topics_i, id2word = dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model = model, texts = text, dictionary = dictionary, coherence = 'c_v')
        coherence_values.append(coherencemodel.get_coherence())
    # the return value will be a coherence graph.

    plt.plot(range(min_topics, max_topics+1), coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.title("Coherence Values")
    plt.legend(("coherence_values"), loc = 'best')
    plt.show()



