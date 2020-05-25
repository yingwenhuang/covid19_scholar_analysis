import re
import os
import pandas
import json
import numpy
import string
import nltk
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.externals import joblib

path = 'temp3/comm_use_subset_pdf_vocabulary.csv'
with open(path, 'r') as file:
    fileReader = csv.reader(file, delimiter = ';')
    dictionary = numpy.array(list(fileReader)).astype(str)
    
# contruct a word-count matrix
# get a list of all files in the certain folder
# path may need to be changed
path = 'temp3/comm_use_subset_pdf/'
file_list = [f for f in os.listdir(path) if f.endswith('.csv')]

matrix = csr_matrix([len(dictionary), len(file_list)])
i = 0
f_list = numpy.zeros(2)

for index, f in enumerate(file_list):
    temp = numpy.array([i, f])
    print(temp)
    f_list = numpy.r_[f_list, temp.T]
    with open(os.path.join(path, f),'r') as file:
        fileReader = csv.reader(file, delimiter = ';')
        data = numpy.array(list(fileReader)).astype(str)

    unique, counts = numpy.unique(data, return_counts = True)
    d = dict(zip(unique, counts))
    for key, value in d.items():
        j = numpy.where(dictionary == key)
        matrix[j[0], i] = value

    i += 1

print(matrix)

f_list = f_list[2:]
l = len(f_list)/2
l = int(l)
f_list = f_list.reshape([l, 2])

# index need to be changed 
numpy.savetxt('comm_use_subset_pdf_index.csv', f_list, fmt = '%s', delimiter = ';')

# transfer to Tfidf matrix
transformer = TfidfTransformer()
matrix_tfidf = transformer.fit_transform(matrix_csr)
# k-means cluster: k in range(1,20)
for k in range(1, 20):
    # set up
    km = KMeans(n_clusters = k)
    # fit model
    km.fit(matrix_tfidf)
    # save model
    model_name = 'k' + str(k) + '.m'
    joblib.dump(km, model_name)
