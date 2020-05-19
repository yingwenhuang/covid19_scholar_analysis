# combine all cvs files to obtain a vocabulary table
import os
import numpy
import string
import csv

# read every json file in the folder

dir_path = 'cleaned_file/noncomm_use_subset/noncomm_use_subset_pmc/noncomm_use_subset_pmc'
data = []
for i in range(0,3):
    subdir_path = dir_path + str(i)
    print(subdir_path)
    file = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
    for r in file:
        print(r)
        path = subdir_path+'/'
        path = path + r
        with open(path, 'r') as file:
            fileReader = csv.reader(file, delimiter = ';')
            data1 = numpy.array(list(fileReader)).astype(str)

        data = numpy.append(data, data1)
        data = numpy.unique(data)

numpy.savetxt("noncomm_use_subset_pmc_vocabulary.csv", data, fmt = "%s", delimiter = ';')

