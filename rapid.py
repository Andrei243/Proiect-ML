# coding: utf-8
import os
import csv
from sklearn import preprocessing
from sklearn import svm

csv_writer = csv.writer(open('submission.csv'))

trainData = []
trainPath = "train/"
# load train data
for i in range(10000, 24000):
    if os.path.isfile(trainPath+str(i)+'.csv'):
        element_act = []
        with open(trainPath + str(i) + '.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            row_count=0
            for row in csv_reader:
                element_act.append(float(row[0]))
                element_act.append(float(row[1]))
                element_act.append(float(row[2]))
                row_count+=1
            while row_count < 150:
                element_act.append(element_act[len(element_act)-4])
                element_act.append(element_act[len(element_act)-4])
                element_act.append(element_act[len(element_act)-4])
                row_count += 1
            while row_count > 150:
                element_act.pop(len(element_act) - 1)
                element_act.pop(len(element_act) - 1)
                element_act.pop(len(element_act) - 1)
                row_count-=1
        trainData.append(element_act)

testData = []
testPath = 'test/'
# load test data
for i in range(10000, 24001):
    if os.path.isfile(testPath + str(i) + '.csv'):
        element_act = []
        with open(testPath + str(i) + '.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            row_count=0
            for row in csv_reader:
                element_act.append(float(row[0]))
                element_act.append(float(row[1]))
                element_act.append(float(row[2]))
                row_count+=1
            while row_count < 150:
                element_act.append(element_act[len(element_act)-4])
                element_act.append(element_act[len(element_act)-4])
                element_act.append(element_act[len(element_act)-4])
                row_count += 1
            while row_count > 150:
                element_act.pop(len(element_act)-1)
                element_act.pop(len(element_act)-1)
                element_act.pop(len(element_act)-1)
                row_count-=1
        testData.append(element_act)

train_labels = []

with open('train_labels.csv') as csv_file:
    csv_reader=csv.reader(csv_file, delimiter=',')
    primul_rand = True
    for row in csv_reader:
        if primul_rand:
            primul_rand = False
        else:
            train_labels.append(int(float(row[1])))

types = [None, "standardized", "min_max", "l1", "l2"]


def normalize_data(train_data, test_data, typed=None):
    if typed is None :
        return train_data, test_data
    if typed == "standardized":
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)
        train_data_scaled=scaler.transform(train_data)
        test_data_scaled=scaler.transform(test_data)
        return train_data_scaled, test_data_scaled
    if typed == "min_max":
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(train_data)
        train_data_scaled=scaler.transform(train_data)
        test_data_scaled=scaler.transform(test_data)
        return train_data_scaled, test_data_scaled
    if typed == "l1":
        train_data_l1=preprocessing.normalize(train_data, "l1")
        test_data_l1=preprocessing.normalize(test_data,"l1")
        return train_data_l1, test_data_l1
    if typed == "l2":
        train_data_l1=preprocessing.normalize(train_data, "l2")
        test_data_l1=preprocessing.normalize(test_data,"l2")
        return train_data_l1, test_data_l1


def svm_classifier_linear(train_data, train_labelss, test_data, c):
    modelSVM = svm.SVC(c, "linear")
    modelSVM.fit(train_data, train_labelss)
    train_labels_predicted = modelSVM.predict(train_data)
    test_labels_predicted = modelSVM.predict(test_data)
    return train_labels_predicted, test_labels_predicted

def svm_classifier_rbf(train_data, train_labelss, test_data, c):
    modelSVM = svm.SVC(c, "linear")
    modelSVM.fit(train_data, train_labelss)
    train_labels_predicted = modelSVM.predict(train_data)
    test_labels_predicted = modelSVM.predict(test_data)
    return train_labels_predicted, test_labels_predicted


def compute_accuracy(true_labels, predicted_labels):
    return (true_labels == predicted_labels).mean()


Cs = [1e-8, 1e-7, 1e-6, 1e-5, 1.5e-5, 1e-4, 1e-2, 1e-1, 1, 10]
bestprob = 0
bestC = 0
linbest=True
trainData, testData = normalize_data(trainData, testData, "standardized")
trainData, testData = normalize_data(trainData, testData, "min_max")
# trainData, testData = normalize_data(trainData, testData, "l1")
trainData, testData = normalize_data(trainData, testData, "l2")
# print("Urmeaza verificarea\n")
# for C in Cs:
#     train_labels_predictedd, test_labels_predictedd = svm_classifier_linear(trainData, train_labels, testData, C)
#     prob = compute_accuracy(train_labels, train_labels_predictedd)
#     print("linear "+str(prob))
#     if(abs(9-prob)<abs(9-bestprob)):
#         bestprob = prob
#         bestC = C
#
# for C in Cs:
#     train_labels_predictedd, test_labels_predictedd = svm_classifier_rbf(trainData, train_labels, testData, C)
#     prob = compute_accuracy(train_labels, train_labels_predictedd)
#     print("linear "+str(prob))
#     if(abs(9-prob)<abs(9-bestprob)):
#         bestprob = prob
#         bestC = C
#         linbest=False


bestres=None
# if linbest:
#     train_labels_predictedd,bestres=svm_classifier_linear(trainData,train_labels,testData,bestC)
# else:
#     train_labels_predictedd,bestres=svm_classifier_rbf(trainData,train_labels,testData,bestC)
train_labels_predictedd,bestres=svm_classifier_rbf(trainData,train_labels,testData,1000)
submisie=open("submission.csv","w")
submisie.write("id,class\n")
print(str(compute_accuracy(train_labels,train_labels_predictedd)))
nr_linie = 0
for i in range(10000, 24001):
    if os.path.isfile(testPath + str(i) + '.csv'):
        submisie.write(str(i)+','+str(bestres[nr_linie])+"\n")
        nr_linie = nr_linie+1
