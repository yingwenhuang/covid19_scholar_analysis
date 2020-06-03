import re
import os
import pandas
import json
import numpy
import string
import nltk
import csv
import sklearn
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
#from sklearn.externals import joblib

# construct a dictionary vector
# path may need to be changed
path = 'covid19_schorlar_analysis/cleaned_file/custom_license/custom_license_pdf_vocabulary.csv'
with open(path, 'r') as file:
    fileReader = csv.reader(file, delimiter = ';')
    dictionary = numpy.array(list(fileReader)).astype(str)

# contruct a word-count matrix
# get a list of all files in the certain folder
# path may need to be changed
i = 0
f_list = numpy.zeros(2)
matrix = csr_matrix(numpy.zeros(len(dictionary)))
for a in range(0,33):
	path = 'covid19_schorlar_analysis/cleaned_file/custom_license/custom_license_pdf/custom_license_pdf'
	path = path + str(a)
	file_list = [f for f in os.listdir(path) if f.endswith('.csv')]
	# read every file to complete word counting
	for index, f in enumerate(file_list):
    		temp = numpy.array([i, f])
    		print(temp)
    		f_list = numpy.r_[f_list, temp.T]
    		with open(os.path.join(path, f),'r') as file:
        		fileReader = csv.reader(file, delimiter = ';')
        		data = numpy.array(list(fileReader)).astype(str)

    		count = numpy.zeros(len(dictionary))
    		for word in data:
        		j = numpy.where(dictionary == word)
        		count[j[0]] += 1

    		matrix = numpy.c_[matrix, count]
    		i += 1

print(matrix)

matrix = matrix[:, 1:]
matrix = matrix.T

f_list = f_list[2:]
l = len(f_list)/2
l = int(l)
f_list = f_list.reshape([l, 2])

# index need to be changed 
numpy.savetxt('custom_license_pdf_index.csv', f_list, fmt = '%s', delimiter = ';')

# transfer to sparse matrix
matrix_csr = csr_matrix(matrix)
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
