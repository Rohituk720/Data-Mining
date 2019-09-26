import sys,time,math
from pyspark import SparkContext
from operator import add
from pyspark.mllib.recommendation import ALS, Rating


def modelbased(inputfile,valfile,outputfile):

    sc = SparkContext(appName="Task2.1")
    start = time.time()

    data = sc.textFile(inputfile)
    data_header = data.first()
    input_data_final = data.filter(lambda rec: rec != data_header).map(lambda string_record: (string_record.split(',')))
    userdata = input_data_final.map(lambda x: x[0]).collect()
    businessdata = input_data_final.map(lambda x: x[1]).collect()

    usermap = {}
    businessmap = {}

    reverseusermap = {}
    reversebusinessmap = {}

    for idx, user in enumerate(userdata):
        usermap[user] = idx
        reverseusermap[idx] = user
        # idx+=1

    for idx, business in enumerate(businessdata):
        businessmap[business] = idx
        reversebusinessmap[idx] = business
        # idx+=1

    ratings = input_data_final.map(lambda x: Rating(int(usermap[x[0]]), int(businessmap[x[1]]), float(x[2])))
    rank = 2
    numIterations = 5
    model = ALS.train(ratings, rank, numIterations)
    test1 = sc.textFile(valfile)
    test_data_header = test1.first()
    testRDD = test1.filter(lambda rec: rec != test_data_header).map(lambda string_record: (string_record.split(',')))
    test_user = testRDD.map(lambda x: x[0]).collect()
    test_business = testRDD.map(lambda x: x[1]).collect()

    for newIdx, user in enumerate(test_user):
        if user not in usermap:
            while newIdx in usermap.values():
                newIdx += 1
            usermap[user] = newIdx
            reverseusermap[newIdx] = user

    for newIdx, business in enumerate(test_business):
        if business not in businessmap:
            while newIdx in businessmap.values():
                newIdx += 1
            businessmap[business] = newIdx
            reversebusinessmap[newIdx] = business

    testingRDD = testRDD.map(lambda x: Rating(int(usermap[x[0]]), int(businessmap[x[1]]), float(x[2])))
    testing_data = testingRDD.map(lambda x: (x[0], x[1]))
    testprediction = model.predictAll(testing_data).map(lambda x: ((x[0], x[1]), x[2])).cache()
    predictions = testprediction.map(lambda x: (x[0][0], x[0][1], x[1])).collect()
    file = open(outputfile, "w")
    file.write('user_id, business_id, prediction\n')
    for pred in predictions:
        file.write(
            str(reverseusermap[pred[0]]) + "," + str(reversebusinessmap[pred[1]]) + "," + str(pred[2]) + "\n")

    file.close()
    end = time.time()
    print("Duration: ", end - start)
    ratesAndPreds = testingRDD.map(lambda r: ((r[0], r[1]), r[2])).join(testprediction)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    print("Mean Squared Error = ", str(MSE ** 0.5))

def userbased(inputfile,valfile,outputfile):

    sc = SparkContext(appName="Task2.2")
    start = time.time()

    traindata = sc.textFile(inputfile)
    traindataheader = traindata.first()
    traindatafinal = traindata.filter(lambda x: x != traindataheader).map(lambda x: (x.split(',')))

    testdata = sc.textFile(valfile)
    testdataheader = testdata.first()
    testdatarest = testdata.filter(lambda x: x != testdataheader)

    ratingsuser = traindatafinal.map(lambda x: ((x[0]), ((x[1]), float(x[2])))).groupByKey().sortByKey(True)
    userratingdict = ratingsuser.mapValues(dict).collectAsMap()
    userratingbc = sc.broadcast(userratingdict)

    ratingbusiness = traindatafinal.map(lambda x: ((x[1]), ((x[0]), float(x[2])))).groupByKey().sortByKey(True)
    businessratingdict = ratingbusiness.mapValues(dict).collectAsMap()
    businessratingbc = sc.broadcast(businessratingdict)

    def getweight(usera, business, businessratingbc, userratingbc):
        business_rating_rdd = businessratingbc.value
        user_rating_rdd = userratingbc.value
        if usera in user_rating_rdd:
            calculatedweight = 0
            userabuslist = list(user_rating_rdd.get(usera))
            userabus = user_rating_rdd.get(usera)
            userarating = sum(userabus.values())
            weightlist = []
            baindex = []
            bbindex = []
            useraavg = userarating / len(userabus)
            if (business_rating_rdd.get(business) == None):
                return (usera, business, str(useraavg))
            else:
                businessUsersList = list(business_rating_rdd.get(business))
                if (len(businessUsersList) != 0):
                    for i in range(0, len(businessUsersList)):
                        totalaratings = 0
                        totalbratings = 0
                        useraindex = 0
                        del baindex[:]
                        del bbindex[:]
                        current_business_value = user_rating_rdd[businessUsersList[i]].get(business)
                        while useraindex < len(userabuslist):
                            if user_rating_rdd[businessUsersList[i]].get(userabuslist[useraindex]):
                                totalaratings += user_rating_rdd[usera].get(userabuslist[useraindex])
                                totalbratings += user_rating_rdd[businessUsersList[i]].get(userabuslist[useraindex])
                                baindex.append(user_rating_rdd[usera].get(userabuslist[useraindex]))
                                bbindex.append(user_rating_rdd[businessUsersList[i]].get(userabuslist[useraindex]))
                            useraindex += 1
                        if len(baindex) != 0:
                            avgusera = totalaratings / len(baindex)
                            avguserb = totalbratings / len(bbindex)
                            numerator = 0
                            denominatorasq = 0
                            denominatorbsq = 0
                            for i in range(0, len(baindex)):
                                ba = baindex[i] - avgusera
                                bb = bbindex[i] - avguserb
                                numerator += (ba) * (bb)
                                denominatorasq += pow(ba, 2)
                                denominatorbsq += pow(bb, 2)
                            denominator = math.sqrt(denominatorasq) * math.sqrt(denominatorbsq)
                            if (denominator != 0):
                                calculatedweight = numerator / denominator
                            predictionDifferenceWeight = (current_business_value - avguserb) * calculatedweight
                            weightlist.append((predictionDifferenceWeight, calculatedweight))
                    predictionnum = 0
                    predictionden = 0
                    for i in range(0, len(weightlist)):
                        predictionnum += weightlist[i][0]
                        predictionden += abs(weightlist[i][1])
                    prediction = -1
                    if (predictionnum == 0 or predictionden == 0):
                        prediction = useraavg
                        return (usera, business, str(prediction))
                    else:
                        prediction = useraavg + (predictionnum / predictionden)
                        if (prediction < 0):
                            prediction = 0.0
                        elif (prediction > 5):
                            prediction = 5.0
                        return (usera, business, str(useraavg))
                else:
                    return (usera, business, "2.7")
        else:
            return (usera, business, str("2.7"))

    testdatafinal = testdatarest.map(lambda x: x.split(","))
    weights = testdatafinal.map(lambda x: getweight(x[0], x[1], businessratingbc, userratingbc))
    nw = weights.collect()
    file = open(outputfile, 'w')
    file.write("user_id, business_id, prediction" + "\n")
    for i in range(0, len(nw)):
        file.write(str(nw[i][0]) + "," + str(nw[i][1]) + "," + str(nw[i][2]) + "\n")
    file.close()
    end = time.time()
    print("Duration: ", end - start)
    #fSplitValues= weights.map(lambda x: (((x[0]), (x[1])), float(x[2])))
    #testDataSplitValues = testdatafinal.map(lambda x: (((x[0]), (x[1])), float(x[2])))
    #joinTestData = testDataSplitValues.join(fSplitValues).map(lambda x: (((x[0]), (x[1])), abs(x[1][0] - x[1][1])))
    #rdd1 = joinTestData.map(lambda x: x[1] ** 2).reduce(lambda x, y: x + y)
    #rmse = math.sqrt(rdd1 / fSplitValues.count())
    #print("RMSE: ", rmse)


def itembased(inputfile,valfile,outputfile):

    sc = SparkContext(appName="Task2.3")
    start = time.time()
    data = sc.textFile(inputfile)
    data_header = data.first()
    input_data_final = data.filter(lambda rec: rec != data_header).map(lambda string_record: (string_record.split(',')))
    userdata = input_data_final.map(lambda x: x[0]).collect()
    businessdata = input_data_final.map(lambda x: x[1]).collect()

    usermap = {}
    businessmap = {}

    reverseusermap = {}
    reversebusinessmap = {}

    for idx, user in enumerate(userdata):
        usermap[user] = idx
        reverseusermap[idx] = user
        # idx+=1

    for idx, business in enumerate(businessdata):
        businessmap[business] = idx
        reversebusinessmap[idx] = business
        # idx+=1

    traindata = input_data_final.map(lambda x: (int(usermap[x[0]]), int(businessmap[x[1]]), float(x[2])))
    test1 = sc.textFile(valfile)
    test_data_header = test1.first()
    testdata = test1.filter(lambda rec: rec != test_data_header).map(lambda string_record: (string_record.split(',')))

    test_user = testdata.map(lambda x: x[0]).collect()
    test_business = testdata.map(lambda x: x[1]).collect()

    for newIdx, user in enumerate(test_user):
        if user not in usermap:
            # newIdx += 1
            while newIdx in usermap.values():
                newIdx += 1
            usermap[user] = newIdx
            reverseusermap[newIdx] = user

    for newIdx, business in enumerate(test_business):
        if business not in businessmap:
            while newIdx in businessmap.values():
                newIdx += 1
            businessmap[business] = newIdx
            reversebusinessmap[newIdx] = business
            # newIdx += 1

    test1 = testdata.map(lambda x: (int(usermap[x[0]]), int(businessmap[x[1]]))).distinct()
    ratingsdata = traindata.map(lambda x: ((x[0], x[1]), x[2])).reduceByKey(add).collectAsMap()
    avgratingdata = traindata.map(lambda x: (x[0], [x[2]])).reduceByKey(add)
    itemaverage = traindata.map(lambda x: (x[1], [x[2]])).reduceByKey(add).map(lambda x: (x[0], float(sum(x[1]) / len(x[1])))).collectAsMap()
    average = avgratingdata.map(lambda x: (x[0], float(sum(x[1]) / len(x[1])))).collectAsMap()
    usertoBusiness = traindata.map(lambda x: (x[0], [x[1]])).reduceByKey(add).collectAsMap()
    businesstoUser = traindata.map(lambda x: (x[1], [x[0]])).reduceByKey(add).collectAsMap()

    def predict(au, ab, wl):
        if ab in itemaverage:
            avg = itemaverage[ab]
        elif au in average:
            avg = average[au]
        else:
            avg = 2.5
        if wl == 0:
            return avg
        nitem = len(wl)
        if nitem == 0:
            return avg
        if (nitem == 1) and (wl[0][0] == ab):
            return avg
        top = list()
        down = list()
        for tup in wl:
            business = tup[0]
            key = (au, business)
            if business != ab and key in ratingsdata:
                try:
                    wba = tup[1]
                except:
                    wba = 0
                val = wba * (ratingsdata[(au, business)])
                top.append(val)
                down.append(abs(wba))
        if (sum(down) == 0 or sum(top) == 0):
            return avg
        pred = abs(float(sum(top) / sum(down)))
        return pred

    def pearsoncorelation(ab, ob, users):
        if (len(users) == 0):
            return -3.0
        ratinga = [ratingsdata[(x, ab)] for x in users]
        suma = sum(ratinga)
        lena = len(ratinga)
        ratingb = [ratingsdata[(x, ob)] for x in users]
        sumb = sum(ratingb)
        lenb = len(ratingb)
        if (lena != 0 and lenb != 0):
            avg1 = suma / lena
            avg2 = sumb / lenb
            diff1 = [x - avg1 for x in ratinga]
            diff2 = [x - avg2 for x in ratingb]
            if (len(diff1) != 0 and len(diff2) != 0):
                root1 = pow(sum([x ** 2 for x in diff1]), 0.5)
                root2 = pow(sum([x ** 2 for x in diff2]), 0.5)
                up = sum([diff1[i] * diff2[i] for i in range(min(len(ratinga), len(ratingb)))])
                den = (root1 * root2)
                if den != 0 and up != 0:
                    return up / den
                elif den == 0 or up == 0:
                    return -3.0
        return -3.0

    def similarity(ua, ba):
        similarbus = []
        try:
            aubusiness = usertoBusiness[ua]
        except:
            similarbus.append((ba, 1.0))
            return similarbus

        if (ba not in businesstoUser):
            similarbus.append((ba, 1.0))
            return similarbus
        try:
            ualist = businesstoUser[ba]
        except:
            return None

        if (len(ualist) == 0):
            return None
        for business in aubusiness:
            if ba != business:
                obusers = businesstoUser[business]
                corratedusers = list(set(ualist).intersection(obusers))
                if (len(corratedusers) != 0):
                    similarity = pearsoncorelation(ba, business, corratedusers)
                    if similarity != -3.0:
                        similarbus.append((business, similarity))
                    else:
                        similarbus.append((business, 2.5))
                else:
                    similarbus.append((business, 2.5))
        similarBus = sorted(similarbus, reverse=True)
        return similarBus

    def flatten(row):
        for innerTup in row:
            if hasattr(innerTup, '__iter__') and not isinstance(innerTup, str) and not isinstance(innerTup, list):
                for a in flatten(innerTup):
                    yield a
            else:
                yield innerTup

    topusers = test1.map(lambda x: ((int(x[0]), int(x[1])), similarity(x[0], x[1]))).filter(lambda x: x[1] != None)
    flatdata = topusers.map(lambda row: tuple(flatten(row)))
    result1 = flatdata.map(lambda x: ((x[0], x[1]), predict(x[0], x[1], x[2]))).distinct()
    flatresult = result1.map(lambda row: tuple(flatten(row)))
    result = flatresult.map(lambda x: (reverseusermap[x[0]], reversebusinessmap[x[1]], x[2])).collect()
    file = open(outputfile, "w")
    file.write('user_id, business_id, prediction\n')
    for pred in result:
        file.write(str(pred[0]) + "," + str(pred[1]) + "," + str(pred[2]) + "\n")
    file.close()
    end = time.time()
    print("Duration: ", end - start)
    #testrating = testdata.map(lambda x: (int(usermap[x[0]]), int(businessmap[x[1]]), float(x[2]))).distinct()
    #ratesAndPreds = testrating.map(lambda r: ((r[0], r[1]), r[2])).join(result1).map(lambda x: (x[1][0] - x[1][1]) ** 2).reduce(lambda x1, x2: x1 + x2)
    #avgDiv = testrating.count()
    #RMSE = ((ratesAndPreds / avgDiv) ** 0.5)
    #print("RMSE= ", RMSE)

inputfile = sys.argv[1]
valfile = sys.argv[2]
case = int(sys.argv[3])
outputfile = sys.argv[4]

if(case==1):
    modelbased(inputfile,valfile,outputfile)

elif(case==2):
    userbased(inputfile,valfile,outputfile)

elif(case==3):
    itembased(inputfile,valfile,outputfile)