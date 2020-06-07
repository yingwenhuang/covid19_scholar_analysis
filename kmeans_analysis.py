# load models and plot SSE graph, pca, and TSNE graph to idendify the best K
import numpy
import csv
import sklearn
import pandas
import scipy
import matplotlib
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# load kmeans model: k = 10
km = joblib.load("k8.m")

# load word count matrix and convert it into tfidf matrix
wordCount = scipy.sparse.load_npz('arxiv_pdf_wordCount.npz')
wordCount = wordCount.T
transformer = TfidfTransformer()
matrix_tfidf = transformer.fit_transform(wordCount)

# load vocabulary
with open("arxiv_vocabulary.csv", 'r') as file:
	csvfile = csv.reader(file, delimiter = ';')
	vocabulary = [row for row in csvfile]
	vocabulary = numpy.array(vocabulary)

# load index file
with open("arxiv_index.csv",'r') as file:
	csvfile = csv.reader(file, delimiter = ';')
	file_index = [row for row in csvfile]
	file_index = numpy.array(file_index)

# load labels
clusters = km.labels_.tolist()
labels = numpy.array(km.labels_)

# construct file_labels
file_label = numpy.c_[file_index, labels]
file_label_frame = pandas.DataFrame(file_label, columns = ['index', 'file_id', 'label'])
print("k = 8:")
print("file counts in each cluster:")
print(file_label_frame['label'].value_counts())
print()

# find the top 10 words in each cluster
order_centroids = km.cluster_centers_.argsort()[:,::-1]
for i in range(0,8):
	print("cluster %d top 10 words:" %i, end = '')
	for ind in order_centroids[i, :10]:
		print(str(vocabulary[ind]), end = '  ')
	print()

# PCA
matrix_tfidf_dense = matrix_tfidf.todense()
pca_data = PCA(n_components = 2).fit_transform(matrix_tfidf_dense)
x = pca_data[:,0]
y = pca_data[:,1]
l = len(x)
labels_color_map = {
    0: 'red', 1: 'blue', 2: 'yellow', 3: 'green', 4: 'grey', 5: 'pink', 6: 'purple', 7: 'brown'
}
for i in range(0, l):
    color = labels_color_map[labels[i]]
    plt.scatter(x[i],y[i], c = color)
plt.title("arxiv kmeans k=8 PCA")
plt.show()
plt.clf()

# t-SNE
# t-SNE
tsne_data = TSNE(n_components = 2).fit_transform(matrix_tfidf_dense)
x = tsne_data[:,0]
y = tsne_data[:,1]
for i in range(0, l):
    color = labels_color_map[labels[i]]
    plt.scatter(x[i],y[i], c = color)
plt.title("arxiv kmeans k=8 TSNE")
plt.show()
