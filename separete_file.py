import re
import os
import pandas
import json
import numpy
import string
import nltk
import shutil
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

# read every json file in the folder
path_to_json = 'pdf_json/'
json_files = [jf for jf in os.listdir(path_to_json)]
i = 0
print("before: ")
print(len(json_files))
for index, js in enumerate(json_files):
    shutil.move(os.path.join(path_to_json, js), 'c32')
    i = i+1
    if i == 990:
        break
    
json_files = [jf for jf in os.listdir(path_to_json)]
print("after: ")
print(len(json_files))
json_files = [jf for jf in os.listdir('c32')]
print("new file: ")
print(len(json_files))
