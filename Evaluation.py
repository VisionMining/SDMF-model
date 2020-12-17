#!/usr/bin/python
# coding=utf-8
import numpy as np
from math import log
import collections

def loadTxtData(fileName, types=[int,int,float], sep=','):
    dataSets = []
    with open(fileName) as file:
        for lines in file:
            tmp = lines[:].strip().split(sep)
            line = []
            for i in range(len(types)):
                line.append(types[i](tmp[i]))
            dataSets.append(line)
    return dataSets

def list2dic(listData): 
    dic = collections.defaultdict(dict) 
    for i in listData:
        dic[i[0]][i[1]]=i[2] 
    return dict(dic)

def loadBiData(fileName, n , types=float , sep=','):
    dataSets = []
    with open(fileName) as file:
        for lines in file:
            tmp = lines[:].strip().split(sep)
            line = []
            for i in range(n):
            	line.append(types(tmp[i]))
            dataSets.append(line)
    return dataSets

def writeListData(fileName, listData, type='w'):
    if len(listData) == 0:
        raise('listData is null')
    with open(fileName, type) as output_file:
        for i in listData:
            output_file.writelines(str(i)[1:-1] + '\n')

#计算编码的汉明距离
def calcHdist(b_usercode, b_itemcode, codelen):
	dist = 0
	for i in range(codelen):
		if b_usercode[i] != b_itemcode[i]:
			dist = dist + 1
	return dist

#取出用户距离排序字典和评分字典的值，返回列表
def dict2list(dict):
	dilist = []
	for eachuser_record in dict.values():
		dilist.append(eachuser_record)
	return dilist

# datadict的列表值的成员是字典还需要转成列表
def dictvalues2list(dict):
	datadict_re = dict2list(dict)
	di_list = []
	for i in range(len(datadict_re)):
		di_list.append(datadict_re[i].items())
	return di_list

#根据汉明距离对测试集用户排序（对测试集中已有的item排序），返回一个dict
def sortHdist(data, usercodes,itemcodes, codelen):
	itemcodesT = itemcodes.T
	for record in data:
		user_id = record[0]
		item_id = record[1]
		dist = calcHdist(usercodes[user_id-1], itemcodesT[item_id-1], codelen)
		record[2] = dist
	dictdata = list2dic(data)
	sortdict = {}
	for item in dictdata.items():
		userids = item[0]
		itemids = item[1]
		sort_ratings = sorted(itemids.items(),key=lambda x:x[1])[:10]	
		sortdict[userids] = sort_ratings
	return sortdict
#根据真实距离对测试集用户排序（对测试集中已有的item排序），返回一个dict
def sortdist(data):
	dictdata = list2dic(data)
	sortdict = {}
	for item in dictdata.items():
		userids = item[0]
		itemids = item[1]
		sort_ratings = sorted(itemids.items(),key=lambda x:x[1], reverse = True)[:10]	
		sortdict[userids] = sort_ratings
	return sortdict
#DCG评价指标(所有用户的DCG均值)
def calcDCG5(dist_list, rating_list, user_num):
 	ratings = []
	movieid_big = []
	sum_dcg = 0.0
	for item in dist_list:
		movieid_list = []
		for i in item:
			movieid = i[0]
			movieid_list.append(movieid)
		movieid_big.append(movieid_list)
	for i in range(user_num):
		each_ratings = []
		id_list = movieid_big[i]
		ulist = rating_list[i]
		for j in range(len(id_list)):
			for k in range(len(ulist)):
				if id_list[j] == ulist[k][0]:
					each_ratings.append(ulist[k][1])
		ratings.append(each_ratings)
	for user_rate in ratings:
		user_dcg = 0.0
		for n in range(len(user_rate)):
			each_dcg = (pow(2, user_rate[n]) - 1) / log(n + 2)
			user_dcg = user_dcg + each_dcg
		sum_dcg = sum_dcg + user_dcg
	ave_dcg = sum_dcg / user_num
	return ave_dcg


data = loadTxtData("test_ratings.csv")
data1 = loadTxtData("test_ratings.csv")
datadict = list2dic(data) 
ub_list = loadBiData("b_usercode6.txt", 35) 
usercodes = np.array(ub_list)                                
ib_list = loadBiData("b_itemcode6.txt", 1682)
itemcodes = np.array(ib_list)
sortdict = sortHdist(data, usercodes,itemcodes, 35)
realsortdict = sortdist(data1)                                                       
user_num = len(sortdict.keys())
dist_list = dict2list(sortdict)
realdist_list = dict2list(realsortdict)
rating_list = dictvalues2list(datadict)
dcg5 = calcDCG5(dist_list, rating_list, user_num)
maxdcg5 = calcDCG5(realdist_list, rating_list, user_num)
ndcg5 = dcg5 / maxdcg5
print ndcg5
