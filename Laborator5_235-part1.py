# coding: utf-8
import os
import csv
from sklearn import preprocessing
from sklearn import svm
import statistics as st
import numpy as np
from sklearn.model_selection import KFold

csv_writer = csv.writer(open('submission.csv'))


def imparte_vectori(xs):
    vector = []
    el = []
    # print(len(xs))
    for i in range(0,134):
        if i % 20 == 0 and i != 0:
            vector.append(el)
            el=[]
        el.append(xs[i])
    for i in range(134, len(xs)):
        el.append(xs[i])
    vector.append(el)
    return vector

def matriceconfuziegoala():
    mat=[]
    for i in range(21):
        el=[]
        for j in range(21):
            el.append(0)
        mat.append(el)
    return mat


def matriceConfuzie(trueLabels,predictedLabels):
    mat = matriceconfuziegoala()
    for i in range(len(trueLabels)):
        mat[predictedLabels[i]][trueLabels[i]]+=1
    return mat

def returneaza_statistici_importante(Xs):
    date = []
    date.append(st.mean(Xs))
    date.append(min(Xs))
    date.append(np.quantile(Xs,0.25))
    date.append(st.median(Xs))
    date.append(np.quantile(Xs,0.75))
    date.append(max(Xs))
    date.append(st.pstdev(Xs))
    date.append(st.pvariance(Xs))
    date.append(st.stdev(Xs))
    date.append(st.variance(Xs))

    return date


def returneaza_vector_important(Xs, Ys, Zs):
    elemente_importante=[]

    vectori=[]
    vectori= vectori + imparte_vectori(Xs) + imparte_vectori(Ys) + imparte_vectori(Zs)
    for vector in vectori:
        elemente_importante=elemente_importante+returneaza_statistici_importante(vector)

    elemente_importante = elemente_importante + returneaza_statistici_importante(Xs)
    elemente_importante = elemente_importante + returneaza_statistici_importante(Ys)
    elemente_importante = elemente_importante + returneaza_statistici_importante(Zs)
    return elemente_importante


minrow=500
maxrow=0
trainData = []
trainPath = "train/"
# load train data
for i in range(10000, 24000):
    if os.path.isfile(trainPath+str(i)+'.csv'):
        element_act = []
        with open(trainPath + str(i) + '.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            row_count=0
            Xs = []
            Ys = []
            Zs = []
            for row in csv_reader:
                Xs.append(float(row[0]))
                Ys.append(float(row[1]))
                Zs.append(float(row[2]))
                row_count += 1
            minrow=min(row_count, minrow)
            maxrow=max(row_count, maxrow)
        trainData.append(returneaza_vector_important(Xs, Ys, Zs))
print("Gata traindata")
testData = []
testPath = 'test/'
# load test data
for i in range(10000, 24001):
    if os.path.isfile(testPath + str(i) + '.csv'):
        element_act = []
        with open(testPath + str(i) + '.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            row_count=0
            Xs = []
            Ys = []
            Zs = []
            for row in csv_reader:
                Xs.append(float(row[0]))
                Ys.append(float(row[1]))
                Zs.append(float(row[2]))
                row_count+=1

        minrow = min(row_count, minrow)
        maxrow = max(row_count, maxrow)
        testData.append(returneaza_vector_important(Xs, Ys, Zs))
print("Gata testdata")
train_labels = []
print("Minrow = " + str(minrow)+" Maxrow = "+str(maxrow))

with open('train_labels.csv') as csv_file:
    csv_reader=csv.reader(csv_file, delimiter=',')
    primul_rand = True
    for row in csv_reader:
        if primul_rand:
            primul_rand = False
        else:
            train_labels.append(int(float(row[1])))


def retvector(vectorinit,pozitii):
    vector=[]
    for i in pozitii:
        vector.append(vectorinit[i])
    return vector

def normalize_data(train_data, test_data, typed=None):
    if typed is None :
        return train_data, test_data
    if typed == "standardized":
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)
        train_data_scaled=scaler.transform(train_data)
        test_data_scaled=scaler.transform(test_data)
        return train_data_scaled, test_data_scaled



def svm_classifier_linear(train_data, train_labelss, test_data, c):
    modelSVM = svm.SVC(c, "linear",gamma='auto')
    modelSVM.fit(train_data, train_labelss)
    train_labels_predicted = modelSVM.predict(train_data)
    test_labels_predicted = modelSVM.predict(test_data)
    return train_labels_predicted, test_labels_predicted


def compute_accuracy(true_labels, predicted_labels):
    return (true_labels == predicted_labels).mean()


trainData,testData=normalize_data(trainData,testData,"standardized")

medprob=0

kf=KFold(n_splits=3)
for train_index, test_index in kf.split(trainData):
    print(train_index)
    print(test_index)
    traintraindata, testtraindata=retvector(trainData,train_index),retvector(trainData,test_index)
    traintrainlabels, testtrainlabels=retvector(train_labels,train_index),retvector(train_labels,test_index)
    train_labels_predictedd, test_labels_predictedd=svm_classifier_linear(traintraindata,traintrainlabels,testtraindata,0.5)
    prob=compute_accuracy(testtrainlabels,test_labels_predictedd)
    print(prob)
    medprob+=prob
    matConfuzie=matriceConfuzie(testtrainlabels,test_labels_predictedd)
    print("    1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20")
    for i in range(1,21):
        string= str(i)
        for j in range(1,21):
            string=string+ str(matConfuzie[i][j])+"  "
        print(string)
    print("")
    print("")
    print("")

medprob=medprob/3
print(str(medprob))


train_labels_predictedd, test_labels_predictedd = svm_classifier_linear(trainData, train_labels, testData, 0.5)
prob = compute_accuracy(train_labels, train_labels_predictedd)
print("linear "+str(prob)+" " + str(0.5))

submisie=open("submission.csv", "w")
submisie.write("id,class\n")
nr_linie = 0
for i in range(10000, 24001):
    if os.path.isfile(testPath + str(i) + '.csv'):
        submisie.write(str(i) + ',' + str(test_labels_predictedd[nr_linie]) + "\n")
        nr_linie = nr_linie + 1
