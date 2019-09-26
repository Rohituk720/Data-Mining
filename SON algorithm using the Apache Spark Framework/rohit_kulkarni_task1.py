import sys,time
from collections import defaultdict
from pyspark import SparkContext
import math

def getCount(allData, candidates):
    itemCountDict = {}
    buckets = list(allData)
    for candidate in candidates:
        if type(candidate) is str:
            candidate = [candidate]
            key = tuple(sorted(candidate))
        else:
            key = candidate
        candidate = set(candidate)
        for basket in buckets:
            if candidate.issubset(basket):
                itemCountDict[key] = itemCountDict.get(key, 0) + 1
    return itemCountDict.items()

def generateFrequents(allData, candidate, support):
    itemCount = defaultdict(int)
    result = defaultdict(str)
    for cand in candidate:
        for basket in allData:
            if set(cand).issubset(set(basket)):
                if support == itemCount[tuple(sorted(set(cand)))]:
                    result[tuple(sorted(set(cand)))] = support
                    break
                else:
                    itemCount[tuple(sorted(set(cand)))] += 1
    return result

def generateCandidatePairs(frequentItems):
    pairs = list()
    for item1 in frequentItems:
        for item2 in frequentItems:
            if item2 > item1:
                pairs.append((item1, item2))
    distinctPairs = list(set(pairs))
    return distinctPairs

def getCandidateItems(frequentItem, size):
    result = list()
    length1 = len(frequentItem) - 1
    length2 = length1 + 1
    for i in range(length1):
        part1 = frequentItem[i]
        for j in range(i + 1, length2):
            part2 = frequentItem[j]
            if part1[0:(size - 2)] == part2[0:(size - 2)]:
                val = list(set(part1).union(set(part2)))
                result.append(val)
            else:
                break
    return result

def apriori(baskets,cutPartition):
    baskets = list(baskets)
    finalOutput = list()
    itemSize1 = dict()
    basketCount = 0
    for basket in baskets:
        for i in basket:
            if i in itemSize1:
                itemSize1[i] = itemSize1[i]+1
            else:
                itemSize1[i] = 1
        basketCount = basketCount+1

    global candidate1
    candidate1 = {}
    candidate1 = {key: val for key, val in itemSize1.items() if val >= cutPartition}
    cutPartition = math.floor(support * (float(basketCount) / float(totalCount)))
    frequent1 = sorted(candidate1)
    finalOutput.extend(frequent1)
    frequentPairs = list(frequent1)
    size = 2
    map(lambda x: not x in frequentPairs and frequentPairs.append(x), frequent1)
    while len(frequentPairs) != 0:
        if(size > 2):
            candidateItems = getCandidateItems(frequentPairs, size)
        elif(size == 2):
            candidateItems = generateCandidatePairs(frequentPairs)

        size += 1
        newPairs = generateFrequents(baskets, candidateItems, cutPartition)
        finalOutput.extend(newPairs)
        frequentPairs = []
        [frequentPairs.append(x) for x in newPairs if x not in frequentPairs]
        frequentPairs.sort()
    return finalOutput

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
case = int(sys.argv[1])
support = float(sys.argv[2])
inputfile = sys.argv[3]
outputfile = sys.argv[4]

sc = SparkContext(appName='Task1')
data = sc.textFile(inputfile)
data_header = data.first()
input_data_rest = data.filter(lambda l: l!= data_header)
input_data_final = input_data_rest.map(lambda l: (l.split(',')))

if(case == 1):
    fulldata = input_data_final.map(lambda x: (x[0], x[1])).combineByKey(lambda v: [v], lambda a, b: a + [b],lambda x, y: x + y)
    fulldata = fulldata.map(lambda r: set(r[1]))
if(case == 2):
    fulldata = input_data_final.map(lambda x: (x[1], x[0])).combineByKey(lambda v: [v], lambda a, b: a + [b],lambda x, y: x + y)
    fulldata = fulldata.map(lambda r: set(r[1]))

partitions = fulldata.getNumPartitions()
totalCount = fulldata.count()
cutPartition = math.floor(support / partitions)

map1 = fulldata.mapPartitions(lambda b: apriori(b, cutPartition)).map(lambda x: (x, 1))
reduce1 = map1.reduceByKey(lambda x, y: x).keys().distinct().collect()
map2 = fulldata.mapPartitions(lambda b: getCount(b, reduce1))
reduce2 = map2.reduceByKey(lambda x, y: x + y)
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