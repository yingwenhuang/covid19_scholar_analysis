# extract the abstract and body-text text from all json files in a folder
# clean up before k-means analysis

import os
import json
import numpy
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# some functions setting up
table = str.maketrans('', '' ,string.punctuation)
porter = PorterStemmer()
d = numpy.empty(1)

# read every json file in the folder
path_to_json = 'CORD-19-research-challenge/arxiv/arxiv/pdf_json/'
json_files = [jf for jf in os.listdir(path_to_json) if jf.endswith('.json')]
i = 0

for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as f:
        data = json.load(f)

    file_name = data["paper_id"]
    cleaned_file = numpy.empty(1)
    i = i+1
    print(i)
    print(file_name)

    # get the text from abstract part
    for p in data["abstract"]:
        # split the sentences into words
        words = word_tokenize(p["text"])
        # lower case
        words = [w.lower() for w in words]
        # remove punctuation from each word
        words = [w.translate(table) for w in words]
        # remove the words not in alphabetic
        words = [w for w in words if w.encode('UTF-8').isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        # transfer to stem words
        words = [porter.stem(w) for w in words]
        # print("body_text: ", words)
        cleaned_file = numpy.append(cleaned_file, words)
        d = numpy.append(d, words)
        d = numpy.unique(d)
        
    # get the text from body_text part
    for p in data["body_text"]:
        # split the sentences into words
        words = word_tokenize(p["text"])
        # lower case
        words = [w.lower() for w in words]
        # remove punctuation from each word
        words = [w.translate(table) for w in words]
        # remove the words not in alphabetic
        words = [w for w in words if w.encode('UTF-8').isalpha()]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        # transfer to stem words
        words = [porter.stem(w) for w in words]
        # print("body_text: ", words)
        cleaned_file = numpy.append(cleaned_file, words)
        d = numpy.append(d, words)
        d = numpy.unique(d)
    print("\n")
    cleaned_file = cleaned_file[1:]
    file_name = 'cleaned_file/arxiv/arxiv_pdf/'+file_name +'.csv'
    numpy.savetxt(file_name, cleaned_file, fmt = "%s", delimiter = ';')
    
d = d[1:]
numpy.savetxt("cleaned_file/arxiv/arxiv_vocabulary.csv", d, fmt = "%s", delimiter = ';')
