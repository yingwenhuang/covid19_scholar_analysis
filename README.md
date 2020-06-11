# data analysis of scholarly articles about COVID19 #
 A data mining project for spark-ai-summit hackthon 
https://databricks.com/blog/2020/04/22/announcing-spark-ai-summit-hackathon-for-social-good.html

Briefly it's about natural language processing on COVID19-related-articles, where most ordinary steps on NLP are involved, 
from cleaning file, to vocabulary filter, k-means clustering , and dimension reduction visualization.

Data source:

https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

For cleaning file:

https://github.com/yingwenhuang/covid19_scholar_analysis/blob/master/wordCount.py

https://github.com/yingwenhuang/covid19_scholar_analysis/blob/master/cleaning_file.py

https://github.com/yingwenhuang/covid19_scholar_analysis/tree/master/cleaned_file

For vocabulary filter and matrix combination:

https://github.com/yingwenhuang/covid19_scholar_analysis/blob/master/vocabulary.py

https://github.com/yingwenhuang/covid19_scholar_analysis/blob/master/vocabulary.csv

https://github.com/yingwenhuang/covid19_scholar_analysis/blob/master/combine_matrix.py

For K-means clustering:

https://github.com/yingwenhuang/covid19_scholar_analysis/blob/master/kmeans.py

https://github.com/yingwenhuang/covid19_scholar_analysis/tree/master/kmeans_models

For dimension reduction visualization:

SSE: https://github.com/yingwenhuang/covid19_scholar_analysis/blob/master/kmeans_sse.py

PCA and TSNE visualization: https://github.com/yingwenhuang/covid19_scholar_analysis/blob/master/kmeans_analysis.py

Results: https://github.com/yingwenhuang/covid19_scholar_analysis/tree/master/kmeans_analysis

Complete notebook:

https://github.com/yingwenhuang/covid19_scholar_analysis/blob/master/covid19_scholar_report%20-%20Jupyter%20Notebook.pdf
