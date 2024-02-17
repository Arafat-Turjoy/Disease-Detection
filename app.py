# -*- coding: utf-8 -*-
from flask import Flask,request, url_for, redirect, render_template, jsonify,json
from flask_cors import CORS
import pandas as pd
from numpy import *
import operator 
import pickle
import re
import wordninja


with open("files/disease_detection_kmean.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)



CORS(app,resources={r"/*": {"origins": "*"}})




def get__disease_data():
    data = pd.read_csv("files/Training.csv")
    data.drop(columns=["Unnamed: 133","prognosis"],inplace=True)
    return data


    
def classify0(inX,k,data = get__disease_data()):
    with open('files/data.pkl', 'rb') as file:
            value = pickle.load(file)
    inX.extend(value)
    print(inX)        
    allocated_array = []
    for i in range(132):
        allocated_array.append(0)
    input_features = [data.columns.get_loc(c) for c in inX if c in data]
    for i in range(0,132):
        if(i in input_features):
            allocated_array[i] = 1
    cluster_number = model.predict([allocated_array])[0]+1
    cluster = pd.read_csv('files/cluster'+ str(cluster_number) +'.csv')
#     cluster.drop(columns=["Unnamed: 133","Unnamed: 0"],inplace=True)
    dataset = cluster.drop(columns=["prognosis"])
    labels = cluster.iloc[:,-1]
    dataSetSize = dataset.shape[0]
    diffMat = tile(allocated_array, (dataSetSize,1)) - dataset
    # print(diffMat)
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
#     print(type(sortedDistIndicies))
    classCount={}
    for i in range(k):
#         print(sortedDistIndicies[i])
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),
    key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

@app.route('/recommendation',methods=['POST']) 

def recommendation():
    random_input = request.get_json()
    with open('files/data.pkl', 'wb') as file:
        pickle.dump(random_input, file)
    substr = []
    if(len(random_input)!=0):
        for item in random_input:
            # print(item)
            res = [item[i: j] for i in range(len(item))
            for j in range(i + 1, len(item) + 1)]
            substr.append(res)

    # print(substr)
    # print(res[2] in item)
    data = get__disease_data()
    # print(data.columns)
    filtered_columns = [column for column in data.columns for sub in substr if any(substr in column for substr in sub) 
                    and len(sub) >= 3 and sub[2] in column]
    filtered_list = [str(item).replace('_', '') for item in filtered_columns]
    input_features = [item for item in filtered_list if item not in random_input]
    final_features = ["_".join(wordninja.split(word)) for word in input_features]
    
   
    
    if(len(input_features)!=0):
        # result = classify0(input_features,5)
        x = final_features
    else:
        x = 'please tell your symptoms correctly'
    return jsonify({'x': x})

@app.route('/prediction',methods=['POST'])

def prediction():
    random_input = request.get_json()
  
    
    if(len(random_input)!=0):
        result = classify0(random_input,5)
        x = f'I think you may have {result}'
    else:
        x = 'please tell your symptoms correctly'
    return x
    

if __name__ == '__main__':
    app.run(debug=True)
