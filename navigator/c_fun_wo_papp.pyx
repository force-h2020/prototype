import numpy as np
import csv

a = np.zeros(10)

def clip(a, min_value, max_value):
    return min(max(a, min_value), max_value)

def read_data(filename, int count_params, int count_entries):
    p = np.zeros([count_params, count_entries], dtype=np.float)
    cdef int i = 0
    with open(filename) as data:
        reader = csv.reader(data, delimiter=",", skipinitialspace=True)
        #next(reader)
        for row in reader:
            for j in range(count_params):
                p[j, i] = float(row[j])
            i = i + 1
    return p

def search_one(valuearray, paramarray, unrestr):
    cdef long i
    for i in unrestr:
        bl = True
        for j in range(valuearray.shape[0]):
            if valuearray[j] != paramarray[j, i]:
                bl = False
        if bl:
            return i
    return -1

def search_all(valuearray, paramarray, unrestr):
    cdef long i
    res = []
    for i in unrestr:
        bl = True
        for j in range(len(valuearray)):
            if valuearray[j] != paramarray[j, i]:
                bl = False
        if bl:
            res.append(i)
    return res

def filter_restricted_values(ar1, ar2, b, t, it):
    cdef int i = 0
    cdef int c = 0
    cdef float tmpx
    cdef float tmpy
    cdef int j
    while(i < ar1.size):
        if ar1[i] > t or ar1[i] < b:
            j = i
            tmpy = ar2[i]
            tmpx = ar1[i]
            while(j < ar1.size - c - 1):
                ar2[j] = ar2[j+1]
                ar1[j] = ar1[j+1]
                j = j + 1
            ar2[j] = tmpy
            ar1[j] = tmpx
            c = c + 1
            i = i - 1
        i = i + 1
        if (c + i) >= it:
            break
    it = it - c - 1
    return it

def fuse_arrays(ar1, ar2):
    cdef int mx = ar1.size
    cdef int i
    a = np.zeros([mx, 2], dtype=np.float)
    for i in range(mx):
        a[i][0] = ar1[i]
        a[i][1] = ar2[i]
    return a
