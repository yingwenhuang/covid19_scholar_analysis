# construct the word-count matrix and indexed file list

import os
import numpy
import string
import scipy
import csv
import sklearn
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.externals import joblib

path = 'cleaned_file/custom_license/custom_license_pdf_vocabulary.csv'
with open(path, 'r') as file:
    fileReader = csv.reader(file, delimiter = ';')
    dictionary = numpy.array(list(fileReader)).astype(str)
    
path = 'cleaned_file/custom_license/custom_license_pdf/custom_license_pdf30/'
file_list = [f for f in os.listdir(path) if f.endswith('.csv')]

matrix = lil_matrix((len(dictionary), len(file_list)))
i = 29480
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
        matrix[j[0], i-29480] = value

    i += 1

print(matrix)
matrix = csr_matrix(matrix)
scipy.sparse.save_npz('custom_license_pdf30_wordCount.npz', matrix)

f_list = f_list[2:]
l = len(f_list)/2
l = int(l)
f_list = f_list.reshape([l, 2])
numpy.savetxt('custom_license_pdf30_index.csv', f_list, fmt = '%s', delimiter = ';')

