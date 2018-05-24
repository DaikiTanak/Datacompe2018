import pandas as pd
import numpy as np

fin = open('data_train.csv')

featname = fin.readline().strip().split(',')
#Load the maximum index
maxID = {}
finID = open('maxID.txt')
for line in finID:
    i,id = line.strip().split('\t')
    maxID[featname[int(i)]] = int(id) + 1


train = pd.read_csv("data_train.csv")
print(train)
