import sys
from pyspark import SparkContext
import time

def minhashing(x):
    global a
    global b
    minNum = [min((ax * x + 1) % b for x in x[1]) for ax in a]
    return (x[0], minNum)

def intermediatestep(x):
    global index1
    global row_per_band
    bands = int(len(x[1]) / row_per_band)
    index1 = index1 + 1
    b_id = x[0]
    signatures_list = x[1]
    bands_list = []
    rowindex = 0
    for b in range(0, bands):
        row = []
        for r in range(0, row_per_band):
            row.append(signatures_list[rowindex])
            rowindex = rowindex + 1
        bands_list.append(((b, tuple(row)), [b_id]))
        row.clear()
    return bands_list

def getCandidateItems(x):
    businesses = x[1]
    businesses.sort()
    candidates = []
    for i in range(0, len(businesses)):
        for j in range(i + 1, len(businesses)):
            if (j > i):
                candidates.append(((businesses[i], businesses[j]), 1))
    return candidates

def jaccardsimilarity(x):
    business1 = x[0][0]
    business2 = x[0][1]
    users1 = set(businessusers[business1])
    users2 = set(businessusers[business2])
    js = float(len(users1.intersection(users2)) / len(users1.union(users2)))
    return (((business1, business2), js))

def index_businesses(x):
    global index2
    index2 = index2 + 1
    return ((x[0], index2))

def outputtofile():
    file = open(outputfile, 'w')
    final_values = []
    file.write("business_id_1, business_id_2, similarity")
    file.write("\n")
    for i in sorteddata.collect():
        if i[0] < i[1][0]:
            final_values.append([i[0],i[1][0],str(i[1][1])])
        else:
            final_values.append([i[1][0],i[0],str(i[1][1])])
    sort_values = sorted(final_values)
    for triples in sort_values:
        file.write(str(triples[0]) + ", " + str(triples[1]) + ", " + str(triples[2]) + "\n")
    file.close()

start = time.time()
sc = SparkContext(appName="task1")
inputfile = sys.argv[1]
outputfile = sys.argv[2]
index = -1
index1 = -1
index2 = -1
row_per_band = 2
data = sc.textFile(inputfile)
input_data_rest = data.map(lambda x: x.split(','))
input_data_final = input_data_rest.filter(lambda x: x[0] != "user_id").persist()
usersrdd = input_data_final.map(lambda a: a[0]).distinct()
businessesrdd = input_data_final.map(lambda a: a[1]).distinct()
businessusers = input_data_final.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y).collectAsMap()

users = usersrdd.collect()
businesses = businessesrdd.collect()

nrows = len(users)
ncols = len(businesses)

usersdict = {}
for u in range(0, nrows):
    usersdict[users[u]] = u

businessesdict = {}
for b in range(0, ncols):
    businessesdict[businesses[b]] = b

characteristicmatrix = input_data_final.map(lambda x: (x[1], [usersdict[x[0]]])).reduceByKey(lambda x, y: x + y)

a = [1, 3, 9, 11, 13, 17, 19, 27, 29, 31, 33, 37, 39, 41, 43, 47, 51, 53, 57, 59, 61, 63, 65, 67, 69, 71, 73, 77, 79,
     81, 83, 85, 87, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131,
     133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177,
     179, 181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215, 217, 219, 221, 223,
     225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 269, 279, 289, 299]
b = nrows
num_of_hash_functions = 30
signaturematrix = characteristicmatrix.map(lambda x: minhashing(x))
signature = signaturematrix.flatMap(lambda x: intermediatestep(x))
candidatepairs = signature.reduceByKey(lambda x, y: x + y).filter(lambda x: len(x[1]) > 1)
candidates = candidatepairs.flatMap(lambda x: getCandidateItems(x)).distinct()
jaccard_similarity = candidates.map(lambda x: jaccardsimilarity(x)).filter(lambda x: x[1] >= 0.5)
sorteddata = jaccard_similarity.map(lambda x: (x[0][1], (x[0][0], x[1]))).sortByKey().map(lambda x: (x[1][0], (x[0], x[1][1]))).sortByKey()

outputtofile()
end = time.time()
print("Duration: " + str(end - start))
