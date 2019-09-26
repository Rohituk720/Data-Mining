import sys
from pyspark import SparkContext
from collections import defaultdict
import time
import itertools
from queue import *

def getpairs(x):
    bid = x[0]
    uid = x[1]
    result = []

    pairs = itertools.combinations(uid, 2)
    for i in pairs:
        i = sorted(i)
        result.append(((i[0], i[1]), [bid]))

    return result

def edgedetection(x, list_edges):
    result = []
    for i in list_edges:
        if (i[0] == x):
            result.append(i[1])
        elif (i[1] == x):
            result.append(i[0])
    result = list(set(result))
    return result

def betweenesscalculation(rnode, avertices, nvertices):
    q = Queue(maxsize=nvertices)
    visited = []
    levels = {}
    parents = {}
    weights = {}

    q.put(rnode)
    visited.append(rnode)
    levels[rnode] = 0
    weights[rnode] = 1

    while (q.empty() != True):
        node = q.get()
        children = avertices[node]

        for i in children:
            if (i not in visited):
                q.put(i)
                parents[i] = [node]
                weights[i] = weights[node]
                visited.append(i)
                levels[i] = levels[node] + 1
            else:
                if (i != rnode):
                    parents[i].append(node)
                    if (levels[node] == levels[i] - 1):
                        weights[i] += weights[node]

    orderv = []
    count = 0
    for i in visited:
        orderv.append((i, count))
        count = count + 1
    reverseorder = sorted(orderv, key=(lambda x: x[1]), reverse=True)
    revorder = []
    nodesvalues = {}
    for i in reverseorder:
        revorder.append(i[0])
        nodesvalues[i[0]] = 1

    betweennessvalues = {}

    for j in revorder:
        if (j != rnode):
            totalweight = 0
            for i in parents[j]:
                if (levels[i] == levels[j] - 1):
                    totalweight += weights[i]

            for i in parents[j]:
                if (levels[i] == levels[j] - 1):
                    source = j
                    dest = i

                    if source < dest:
                        pair = tuple((source, dest))
                    else:
                        pair = tuple((dest, source))

                    if (pair not in betweennessvalues.keys()):
                        betweennessvalues[pair] = float(nodesvalues[source] * weights[dest] / totalweight)
                    else:
                        betweennessvalues[pair] += float(nodesvalues[source] * weights[dest] / totalweight)

                    nodesvalues[dest] += float(nodesvalues[source] * weights[dest] / totalweight)

    betweennesslist = []
    for key, value in betweennessvalues.items():
        temp = [key, value]
        betweennesslist.append(temp)

    return betweennesslist

def bfs(rnode, avertices, nvertices):
    visited = []
    edges = []
    q = Queue(maxsize=nvertices)

    q.put(rnode)
    visited.append(rnode)

    while (q.empty() != True):
        node = q.get()
        children = avertices[node]

        for i in children:
            if (i not in visited):
                q.put(i)
                visited.append(i)

            pair = sorted((node, i))
            if (pair not in edges):
                edges.append(pair)

    return (visited, edges)

def componentdeletion(remaindergraph, component):
    cvertices = component[0]

    for v in cvertices:
        del remaindergraph[v]

    for i in remaindergraph.keys():
        adjlist = remaindergraph[i]
        for v in cvertices:
            if (v in adjlist):
                adjlist.remove(i[1])
        remaindergraph[i] = adjlist

    return remaindergraph

def isNull(avertices):
    if (len(avertices) == 0):
        return True
    else:
        for i in avertices.keys():
            adj_list = avertices[i]
            if (len(adj_list) != 0):
                return False
            else:
                pass
        return True

def getconnectedcomponents(avertices):
    connectedcomponents = []
    remaindergraph = avertices

    while (isNull(remaindergraph) == False):
        vertices = []

        for key, value in remaindergraph.items():
            vertices.append(key)

        vertices = list(set(vertices))
        root = vertices[0]
        compvalues = bfs(root, avertices, len(vertices))
        connectedcomponents.append(compvalues)
        remaindergraph = componentdeletion(remaindergraph, compvalues)

    return connectedcomponents

def modularitycalculation(avertices, connectedcomponents, m):
    modularity = 0
    for c in connectedcomponents:
        cvertices = c[0]
        for i in cvertices:
            for j in cvertices:
                Aij = 0
                adjaclist = avertices[str(i)]
                if (j in adjaclist):
                    Aij = 1

                ki = len(avertices[i])
                kj = len(avertices[j])

                modularity += Aij - (ki * kj) / (2 * m)

    modularity = modularity / (2 * m)
    return modularity

def generateadjacencymatrix(connectedcomponents):
    result = {}
    for c in connectedcomponents:
        cedges = c[1]
        for i in cedges:
            if (i[0] in result.keys()):
                result[i[0]].append(i[1])
            else:
                result[i[0]] = [i[1]]

            if (i[1] in result.keys()):
                result[i[1]].append(i[0])
            else:
                result[i[1]] = [i[0]]

    return result

def edgedeletion(amatrix, firstedge):
    if (firstedge[0] in amatrix.keys()):
        l = amatrix[firstedge[0]]
        if (firstedge[1] in l):
            l.remove(firstedge[1])

    if (firstedge[1] in amatrix.keys()):
        l = amatrix[firstedge[1]]
        if (firstedge[0] in l):
            l.remove(firstedge[0])
    return amatrix

def outputtofile():
    file = open(betweenessfile, 'w')
    ctr = 0
    for i in betweennessdata.collect():
        if (ctr == 0):
            ctr = 1
        else:
            file.write("\n")
        file.write(str(i[0]) + ", " + str(i[1]))
    file.close()

    file = open(communityfile, 'w')
    ctr = 0
    for i in sortedcommunities:
        if (ctr == 0):
            ctr = 1
        else:
            file.write("\n")
        s = str(i[0]).replace("[", "").replace("]", "")
        file.write(s)
    file.close()


start = time.time()
sc = SparkContext()
filterthreshold = int(sys.argv[1])
inputfile = sys.argv[2]
betweenessfile = sys.argv[3]
communityfile = sys.argv[4]

data = sc.textFile(inputfile)
inputdata = data.map(lambda x: x.split(','))
inputdatafinal = inputdata.filter(lambda x: x[0] != "user_id").persist()

businessusers = inputdatafinal.map(lambda x: (x[1], [x[0]])).reduceByKey(lambda x, y: x + y)
pairusers = businessusers.flatMap(lambda x: getpairs(x)).reduceByKey(lambda x, y: x + y).filter(lambda x: len(x[1]) >= filterthreshold).map(lambda x: x[0])

verticesdata = pairusers.flatMap(lambda x: [(x[0]), (x[1])]).distinct()
listvertices = verticesdata.collect()
nvertices = len(listvertices)

edgesdata = pairusers.map(lambda x: (x[0], x[1])).map(lambda x: (x[0], x[1]))
list_edges = edgesdata.collect()
adjacent_vertices = verticesdata.map(lambda x: (x, edgedetection(x, list_edges))).collectAsMap()

betweennessdata = verticesdata.flatMap(lambda x: betweenesscalculation(x, adjacent_vertices, nvertices)) \
    .reduceByKey(lambda x, y: (x + y)).map(lambda x: (x[0], float(x[1] / 2))).sortByKey().map(
    lambda x: (x[1], x[0])).sortByKey(ascending=False).map(lambda x: (x[1], x[0]))

firstedge = betweennessdata.take(1)[0][0]
m = edgesdata.count()

adjacency_matrix = adjacent_vertices.copy()
connected_components = getconnectedcomponents(adjacency_matrix)
modularity = modularitycalculation(adjacent_vertices, connected_components, m)
adjacency_matrix = adjacent_vertices.copy()

highmodularity = -1
communities = []
count = 0
while (1):
    adjacency_matrix = edgedeletion(adjacency_matrix, firstedge)
    connected_components = getconnectedcomponents(adjacency_matrix)
    modularity = modularitycalculation(adjacent_vertices, connected_components, m)
    adjacency_matrix = generateadjacencymatrix(connected_components)
    temp = []
    for i in adjacency_matrix.keys():
        temp.append(i)
    temp = list(set(temp))
    v_rdd = sc.parallelize(temp)
    betweenness_temp = v_rdd.flatMap(lambda x: betweenesscalculation(x, adjacency_matrix, nvertices)) \
        .reduceByKey(lambda x, y: (x + y)).map(lambda x: (x[0], float(x[1] / 2))).sortByKey().map(
        lambda x: (x[1], x[0])).sortByKey(ascending=False).map(lambda x: (x[1], x[0]))
    firstedge = betweenness_temp.take(1)[0][0]

    if (modularity >= highmodularity):
        highmodularity = modularity
        communities = connected_components

    count += 1
    if (count == 50):
        break

sortedcommunities = []
for i in communities:
    item = sorted(i[0])
    sortedcommunities.append((item, len(item)))

sortedcommunities.sort()
sortedcommunities.sort(key=lambda x: x[1])
outputtofile()
end = time.time()
print("Duration: " + str(end - start))