import numpy as np
import scipy as scipy
import lxmls.classifiers.linear_classifier as lc
import sys
from lxmls.distributions.gaussian import *


class MultinomialNaiveBayes(lc.LinearClassifier):

    def __init__(self, xtype="gaussian"):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = False
        self.smooth_param = 1

    def train(self, x, y):
        # n_docs = no. of documents (my samples)
        # n_words = no. of unique words (my vocabulary)
        n_docs, n_words = x.shape

        # classes = a list of possible classes
        classes = np.unique(y)
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]

        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words, n_classes))

        for i in range(n_classes):

            # MEU JEITO DE FAZER!!!
            docs_in_class = []
            for n in range(y.shape[0]):
                if y[n] == classes[i]:
                    docs_in_class.append(n)

            # A SOLUÇÃO DO CURSO
            # docs_in_class, _ = np.nonzero(y == classes[i])  # docs_in_class = indices of documents in class i


            prior[i] = 1.0 * len(docs_in_class) / n_docs  # prior = fraction of documents with this class

            # word_count_in_class = count of word occurrences in documents of class i

            word_count_in_class = [0] * 10

            for n in range(len(docs_in_class)):
                for j in range(x.shape[1]):
                    word_count_in_class[j] += x[docs_in_class[n], j]

            print(word_count_in_class)
            word_count_in_class = x[docs_in_class, :].sum(0)
            print(word_count_in_class)

            # word_count_in_class = []
            #
            #
            # for j in range(len(docs_in_class)):   # iterando na linha
            #     for word in range(x.shape[1]):    # iterando na coluna
            #         word_count_in_class[j] += x[docs_in_class[j],word]

            # word_count_in_class = x[docs_in_class, :].sum(0)

            total_words_in_class = word_count_in_class.sum()  # total_words_in_class = total number of words in documents of class i
            if not self.smooth:
                # likelihood = count of occurrences of a word in a class
                likelihood[:, i] = word_count_in_class / total_words_in_class
            else:
                likelihood[:, i] = (word_count_in_class+self.smooth_param) / (total_words_in_class + self.smooth_param*n_words)


        params = np.zeros((n_words+1, n_classes))
        for i in range(n_classes):
            params[0, i] = np.log(prior[i])
            params[1:, i] = np.nan_to_num(np.log(likelihood[:, i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
