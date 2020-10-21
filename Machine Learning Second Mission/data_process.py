import pandas as pd
from sklearn.model_selection import train_test_split
import random

def read_data():
    with open('data/train_binary.csv','r') as f:
        header = f.readline()
        values = []
        for line in f:
            values.append(line)
    return header, values

def write_data(header, values):
    ran = random.sample(range(0, len(values)), int(len(values)/10))
    with open('data/mini_train_binary.csv','w') as f:
        f.write(header)
        for i in ran:
            f.write(values[i])

if __name__ == '__main__':
    header, values = read_data()
    write_data(header, values)


