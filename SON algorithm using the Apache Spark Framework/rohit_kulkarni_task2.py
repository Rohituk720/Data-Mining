import sys
from pyspark import SparkContext
from collections import Counter
from itertools import combinations
import time

def getCount(allData, candidates):
    itemCountDict = {}
    baskets = list(allData)
    for cand in candidates:
        if type(cand) == str:
            cand = [cand]
            key = tuple(sorted(cand))
        else:
            key = cand
        cand = set(cand)
        for basket in baskets:
            if cand.issubset(basket):
                itemCountDict[key] = itemCountDict.get(key, 0) + 1
    return itemCountDict.items()

def generateCandidatePairs(frequentItem1):
    Size2 = list()
    frequentItem = set(frequentItem1)
    for s in combinations(frequentItem, 2):
        s = list(s)
        s.sort()
        Size2.append(s)
    return Size2

def getPrunedItems(baskets, candidatesItem, support):
    count = {}
    for cand in candidatesItem:
        cand = set(cand)
        sortItems = sorted(cand)
        tupleitems = tuple(sortItems)
        for basket in baskets:
            if cand.issubset(basket):
                count[tupleitems] = count.get(tupleitems, 0) + 1
    frequentItem1 = {}
    for i in count:
        if count[i] >= support:
            frequentItem1[i] = count[i]
    frequentItems = sorted(frequentItem1)
    return frequentItems

def getCandidateItems(frequentItem, size):
    res = list()
    length1 = len(frequentItem) - 1
    length2 = length1 + 1
    for i in range(length1):
        part1 = frequentItem[i]
        for j in range(i + 1, length2):
            part2 = frequentItem[j]
            if part1[0:(size - 2)] == part2[0:(size - 2)]:
                val = list(set(part1).union(set(part2)))
                res.append(val)
            else:
                break
    return res

def apriori(baskets, support, count):
    basketslist = list(baskets)
    size = len(basketslist)
    threshold = float(support) * (float(size) / float(count))
    counts = Counter()
    for basket in basketslist:
        counts.update(basket)
    frequentItem1 = {}
    for i in counts:
        if counts[i] >= threshold:
            frequentItem1[i] = counts[i]
    frequentItem1 = sorted(frequentItem1)
    finalItem = list()
    finalItem.extend(frequentItem1)
    candidateItem2 = generateCandidatePairs(frequentItem1)
    frequentItem2 = getPrunedItems(basketslist, candidateItem2, threshold)
    finalItem.extend(frequentItem2)
    length = 3
    frequentItems = frequentItem2
    while len(frequentItems) != 0:
        candidateItems = getCandidateItems(frequentItems, length)
        frequentItems = getPrunedItems(basketslist, candidateItems, threshold)
        finalItem.extend(frequentItems)
        frequentItems.sort()
        length = length + 1
    return finalItem

def outputforcandidate(candidateItems):
    outfile = open(outputfile, "w")
    outfile.write("Candidates:")
    outfile.write("\n")
    itemSize = len(candidateItems[0])
    for candidate in candidateItems:
        candidateSize = len(candidate)
        if (candidateSize == itemSize and candidate != candidateItems[0]):
            outfile.write(", ")
        elif (candidateSize != 1):
            outfile.write("\n\n")
        if type(candidate) is tuple:
            outfile.write("('")
            for i in range(0, candidateSize - 1):
                outfile.write(str(candidate[i]) + ", ")
            outfile.write(str(candidate[candidateSize - 1]) + "')")
        else:
            outfile.write(str(candidate) + ",")
        itemSize = candidateSize


def outputforfrequent(frequentItems):
    outfile = open(outputfile, "a")
    outfile.write("\n\nFrequent Itemsets:")
    outfile.write("\n")
    itemSize = len(frequentItems[0])
    for frequentItem in frequentItems:
        frequentItemSize = len(frequentItem)
        if (frequentItemSize == itemSize and frequentItem != frequentItems[0]):
            outfile.write(", ")
        elif (frequentItemSize != 1):
            outfile.write("\n\n")
        if type(frequentItem) is tuple:
            outfile.write("('")
            for i in range(0, frequentItemSize - 1):
                outfile.write(str(frequentItem[i]) + ", ")
            outfile.write(str(frequentItem[frequentItemSize - 1]) + "')")
        else:
            outfile.write(str(frequentItem) + ",")
        itemSize = frequentItemSize

start = time.time()
threshold = int(sys.argv[1])
support = float(sys.argv[2])
inputfile = sys.argv[3]
outputfile = sys.argv[4]

sc = SparkContext(appName='Task2')
data = sc.textFile(inputfile)
data_header = data.first()
input_data_rest = data.filter(lambda l: l!= data_header)
input_data_final = input_data_rest.map(lambda l: (l.split(',')))
userDataRdd = input_data_final.map(lambda x: (x[0], x[1])).combineByKey(lambda v: [v], lambda a, b: a + [b],lambda x, y: x + y).filter(lambda x: len(x[1])>threshold).map(lambda r:set(r[1]))
count = userDataRdd.count()

map1 = userDataRdd.mapPartitions(lambda baskets: apriori(baskets, support, count)).map(lambda x: (x, 1))
reduce1 = map1.reduceByKey(lambda x, y: (1)).keys().collect()
map2 = userDataRdd.mapPartitions(lambda baskets: getCount(baskets, reduce1))
reduce2 = map2.reduceByKey(lambda x, y: (x + y))
output = reduce2.filter(lambda x: x[1] >= support)
file = open(outputfile, "w")
candidateItems = reduce2.map(lambda a: a[0]).collect()
candidateItems = sorted(candidateItems, key=lambda i: (len(i), i))
frequentItems = output.map(lambda a: a[0]).collect()
frequentItems = sorted(frequentItems, key=lambda i: (len(i), i))
outputforcandidate(candidateItems)
outputforfrequent(frequentItems)
end = time.time()
print("Duration: ", end - start)