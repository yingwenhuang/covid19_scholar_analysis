# move files from one folder to another one
import os
import numpy
import string
import shutil

# read every file in the folder
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
