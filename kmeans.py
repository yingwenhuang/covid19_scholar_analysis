# kmeans cluster and save model for further analysis
import numpy
import sklearn
import scipy
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.externals import joblib

# read the word count matrix
matrix = scipy.sparse.load_npz("wordCount/comm_use_subset_wordCount/comm_use_subset_pdf_wordCount.npz")

# transfer to Tfidf matrix
transformer = TfidfTransformer()
matrix_tfidf = transformer.fit_transform(matrix)

# k-means cluster: k in range(1,20)
for k in range(1, 20):
    print(k)
    # set up
    km = KMeans(n_clusters = k)
    # fit model
    km.fit(matrix_tfidf)
    # save model
    model_name = 'kmeans_models/comm_use_subset/k' + str(k) + '.m'
    joblib.dump(km, model_name)
