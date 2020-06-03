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
import scipy
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
pdf1 = scipy.sparse.load_npz("pdf1.npz")
pdf2 = scipy.sparse.load_npz("pdf2.npz")
pdf3 = scipy.sparse.load_npz("pdf3.npz")
pdf4 = scipy.sparse.load_npz("pdf4.npz")
pdf5 = scipy.sparse.load_npz("pdf5.npz")
pdf6 = scipy.sparse.load_npz("pdf6.npz")
pdf7 = scipy.sparse.load_npz("pdf7.npz")
pdf8 = scipy.sparse.load_npz("pdf8.npz")
pdf9 = scipy.sparse.load_npz("pdf9.npz")
pdf10 = scipy.sparse.load_npz("pdf10.npz")
pdf11 = scipy.sparse.load_npz("pdf11.npz")
pdf12 = scipy.sparse.load_npz("pdf12.npz")
pdf13 = scipy.sparse.load_npz("pdf13.npz")
pdf14 = scipy.sparse.load_npz("pdf14.npz")
pdf15 = scipy.sparse.load_npz("pdf15.npz")
pdf16 = scipy.sparse.load_npz("pdf16.npz")
pdf17 = scipy.sparse.load_npz("pdf17.npz")
pdf18 = scipy.sparse.load_npz("pdf18.npz")
pdf19 = scipy.sparse.load_npz("pdf19.npz")
pdf20 = scipy.sparse.load_npz("pdf20.npz")
pdf21 = scipy.sparse.load_npz("pdf21.npz")
pdf22 = scipy.sparse.load_npz("pdf22.npz")
pdf23 = scipy.sparse.load_npz("pdf23.npz")
pdf24 = scipy.sparse.load_npz("pdf24.npz")
pdf25 = scipy.sparse.load_npz("pdf25.npz")
pdf26 = scipy.sparse.load_npz("pdf26.npz")
pdf27 = scipy.sparse.load_npz("pdf27.npz")
pdf28 = scipy.sparse.load_npz("pdf28.npz")
pdf29 = scipy.sparse.load_npz("pdf29.npz")
pdf30 = scipy.sparse.load_npz("pdf30.npz")
pdf31 = scipy.sparse.load_npz("pdf31.npz")
pdf32 = scipy.sparse.load_npz("pdf32.npz")

pdf = scipy.sparse.hstack([pdf0, pdf1, pdf2])
pdf = scipy.sparse.hstack([pdf, pdf3, pdf4])
pdf = scipy.sparse.hstack([pdf, pdf5, pdf6])
pdf = scipy.sparse.hstack([pdf, pdf7, pdf8])
pdf = scipy.sparse.hstack([pdf, pdf9, pdf10])
pdf = scipy.sparse.hstack([pdf, pdf11, pdf12])
pdf = scipy.sparse.hstack([pdf, pdf13, pdf14])
pdf = scipy.sparse.hstack([pdf, pdf15, pdf16])
pdf = scipy.sparse.hstack([pdf, pdf17, pdf18])
pdf = scipy.sparse.hstack([pdf, pdf19, pdf20])
pdf = scipy.sparse.hstack([pdf, pdf21, pdf22])
pdf = scipy.sparse.hstack([pdf, pdf23, pdf24])
pdf = scipy.sparse.hstack([pdf, pdf25, pdf26])
pdf = scipy.sparse.hstack([pdf, pdf27, pdf28])
pdf = scipy.sparse.hstack([pdf, pdf29, pdf30])
pdf = scipy.sparse.hstack([pdf, pdf31, pdf32])

# transfer to Tfidf matrix
transformer = TfidfTransformer()
matrix_tfidf = transformer.fit_transform(pdf)
# k-means cluster: k in range(1,20)
for k in range(1, 20):
    # set up
    km = KMeans(n_clusters = k)
    # fit model
    km.fit(matrix_tfidf)
    # save model
    model_name = 'k' + str(k) + '.m'
    joblib.dump(km, model_name)
