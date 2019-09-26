from pyspark import SparkContext
import json
from operator import add
import sys

inputfile = sys.argv[1]
outputfile = sys.argv[2]
output = {}

sc = SparkContext(appName="task1")
data = sc.textFile(inputfile)
data_final = data.map(lambda r: json.loads(r))
data_partition = data_final.repartition(4)
n_review_useful = data_partition.filter(lambda r: r['useful'] > 0).count()
output['n_review_useful'] = n_review_useful
n_review_5_star = data_partition.filter(lambda r: r['stars'] == 5.0).count()
output['n_review_5_star'] = n_review_5_star
n_characters = data_partition.map(lambda r: (len(r['text']), 1)).reduceByKey(add).takeOrdered(1, key=lambda u: (-u[0]))
n_characters_value = [key[0] for key in n_characters]
output['n_characters'] = n_characters_value[0]
n_user = data_partition.map(lambda r: (r['user_id'], 1)).reduceByKey(add).count()
output['n_user'] = n_user
top20_user_collection = data_partition.map(lambda r: (r['user_id'], 1)).reduceByKey(add).takeOrdered(20, key=lambda x: (-x[1], x[0]))
top20_user = list(map(list, top20_user_collection))
output['top20_user'] = top20_user
n_business = data_partition.map(lambda r: (r['business_id'], 1)).reduceByKey(add).count()
output['n_business'] = n_business
top20_business_collection = data_partition.map(lambda r: [r['business_id'], 1]).reduceByKey(add).takeOrdered(20, key=lambda x: (-x[1], x[0]))
top20_business = list(map(list, top20_business_collection))
output['top20_business'] = top20_business

with open(outputfile, 'w') as fileoutput:
    json.dump(output, fileoutput)