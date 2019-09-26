from pyspark import SparkContext
import sys
import json
import time

inputfile1 = sys.argv[1]
inputfile2 = sys.argv[2]
outputfile1 = sys.argv[3]
outputfile2 = sys.argv[4]

sc = SparkContext(appName="task2")
output = {}
data1 = sc.textFile(inputfile1)
input_data1 = data1.map(lambda r: json.loads(r))
data_partition1 = input_data1.repartition(4)

data2 = sc.textFile(inputfile2)
input_data2 = data2.map(lambda s: json.loads(s))
data_partition2 = input_data2.repartition(4)

reviewdata = data_partition1.map(lambda i: [i['business_id'], i['stars']])
businessdata = data_partition2.map(lambda i: [i['business_id'], i['state']])

joinrdd = businessdata.join(reviewdata).map(lambda i: i[1])
firsttup = (0, 0)
average = joinrdd.aggregateByKey(firsttup, lambda i, j: (i[0]+j, i[1]+1), lambda i, j: (i[0]+j[0], i[1]+j[1]))

final_avg = average.map(lambda r: (r[0], r[1][0]/r[1][1])).sortBy(lambda s: (-s[1], s[0]))
final_avg_list = final_avg.collect()

with open(outputfile1, "w") as fileoutput:
    fileoutput.write("state,stars\n")
    for values in final_avg_list:
        fileoutput.write(str(values[0])+','+str(values[1]))
        fileoutput.write("\n")

starttime1 = time.time()
m1_collect = final_avg.map(lambda p: p[0]).collect()
m1_top_5 = m1_collect[:5]
print(m1_top_5)
endtime1 = time.time()
m1 = endtime1 - starttime1
output['m1'] = m1

starttime2 = time.time()
m2_top_5 = sc.parallelize(final_avg.take(5)).map(lambda q: q[0]).collect()
print(m2_top_5)
endtime2 = time.time()
m2 = endtime2-starttime2
output['m2'] = m2

output['explanation'] = "From the time taken it seems the first method runs a little faster compared to the 2nd, but the difference is less than a second, so the conclusion is that the runtime for both methods is almost same"

with open(outputfile2, 'w') as fileoutput:
    json.dump(output, fileoutput)
