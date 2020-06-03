# prepare a final matrix for k-means analysis

import string
import scipy
from scipy import sparse

path = 'custom_license_pdf_wordCount/custom_license_pdf'
path1 = 'custom_license_pdf_wordCount/custom_license_pdf0.npz'
temp = scipy.sparse.load_npz(path1)

for i in range(1,33):
    path1 = path + str(i) + '.npz'
    t = sparse.load_npz(path1)
    temp = sparse.hstack([temp, t])

sparse.save_npz('custom_license_pdf_wordCount.npz', temp)
