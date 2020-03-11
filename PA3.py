#!/usr/bin/env python
# coding: utf-8

import csv
import numpy as np
import random
import math
import sys

def logistic_loss(train_y, pred_y):
    innerProduct = np.multiply(train_y, pred_y)
    log_loss = np.log(1 + np.exp(-innerProduct))
    return np.mean(log_loss)


def hinge_loss(train_y, pred_y):
    innerProduct = np.multiply(train_y, pred_y)
    hinge_loss_vector = np.maximum(0,1-innerProduct)
    return np.mean(hinge_loss_vector)

# Sum of all the values of the weight
def l1_reg(w):
    l1_reg_loss = 0;
    for i in range(1,len(w)):
        l1_reg_loss += abs(w[i])
    return l1_reg_loss

#Dot product of the weight with itself
def l2_reg(w):    
    l2_reg_loss = np.dot(w[1:], np.transpose(w[1:]))
    return l2_reg_loss


def train_classifier(train_x, train_y, learn_rate, loss, lambda_val=None, regularizer=None):
    #Initialize weight with bias
    weight_vector = np.random.rand(len(train_x[0]) + 1) 
    #Iteration
    num_iters = 100
    #Numerical_differentiation, as suggested
    h = 0.0001
    #Run for num_iters times
    for i in range(num_iters):
        
        current_weight = np.copy(weight_vector)
        #Delta_weight for updating: w = w - learn_rate*delta_weight
        delta_weight = np.zeros(len(train_x[0]) + 1) 
        #Prediction using current weight
        predict_y = test_classifier(current_weight,train_x)
        
        #Check for lambda existence, if yes then add regularization
        if(lambda_val):
            current_loss = loss(train_y, predict_y) + lambda_val*regularizer(current_weight)
        else:
            current_loss = loss(train_y, predict_y)
            
        
        for i in range(len(delta_weight)):
            tmp_current_weight = np.copy(current_weight)
            tmp_current_weight[i] = tmp_current_weight[i] + h;
            
            tmp_predict_y = test_classifier(tmp_current_weight,train_x)
            
            # Find delta_weight using loss function
            
            #Check for lambda existence
            if(lambda_val):
                tmp_loss = loss(train_y, tmp_predict_y) + lambda_val*regularizer(tmp_current_weight)
            else:
                tmp_loss = loss(train_y, tmp_predict_y)
            
            #Differentiation
            delta_weight[i] = (tmp_loss - current_loss) / h

        #Update weight
        weight_vector = current_weight - learn_rate*delta_weight
           
    return weight_vector

#Return pred_y as inner product of weights and feature values
def test_classifier(w, test_x):
    pred_y = np.zeros(len(test_x))
    for i in range(len(test_x)):
        pred_y[i] = np.dot(w[1:], test_x[i]) + w[0]
    return pred_y    

def normalize(trainX,testX):
    # Standardize the dataset
    dataX_trans = trainX.transpose()
    column = 0
    for row in dataX_trans:
        #Subtract mean from every value, then divide by deviation
        mean = np.mean(row)
        std = np.std(row)
        for i in range(len(dataX_trans[0])):
            trainX[i][column] -= mean
            trainX[i][column] /= std
        for i in range(len(testX.transpose()[0])):
            testX[i][column] -= mean
            testX[i][column] /= std
        column += 1

#Find accuracy
def compute_accuracy(test_y, pred_y):
    #Convert predicted label into -1 and 1
    convert_pred_y = np.copy(pred_y)
    for j in range(len(convert_pred_y)):
        if(convert_pred_y[j] < 6):
            convert_pred_y[j] = -1
        elif (convert_pred_y[j] > 6):
            convert_pred_y[j] = 1
    
	#Vector filled with booleans
    compare = (test_y == convert_pred_y)
    match_count = 0
    for i in range(len(compare)):
        if (compare[i] == True):
            match_count += 1 
    
    return (match_count/len(test_y))

def main():

    # Read the training data file
    szDatasetPath = 'winequality-white.csv'
    listClasses = []
    listAttrs = []
    bFirstRow = True
    with open(szDatasetPath) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for row in csvReader:
            if bFirstRow:
                bFirstRow = False
                continue
            if int(row[-1]) < 6:
                listClasses.append(-1)
                listAttrs.append(list(map(float, row[1:len(row) - 1])))
            elif int(row[-1]) > 6:
                listClasses.append(+1)
                listAttrs.append(list(map(float, row[1:len(row) - 1])))
    dataX = np.array(listAttrs)
    dataY = np.array(listClasses)
    
    # 5-fold cross-validation
    np.set_printoptions(precision=8)
    np.set_printoptions(suppress=True)
    subsetNum = int((len(dataX)/5))
    
    #Learn rate and lambda value to try out
    learn_rate_array = {0.001,0.0001,0.00001}
    lambda_array = {0.1,0.01,0.001}
    for learnRate in learn_rate_array:
        for lambdaVal in lambda_array:
            
			#Fold 5 times, split the train and test sets
            acc_svm_avg=0
            acc_log_avg=0
            #Initialize 5-folds
            for i in range(5):
                if(i == 0):
                    #Split
                    subdataX = np.split(dataX,[subsetNum])
                    subdataY = np.split(dataY,[subsetNum])
                    #Train
                    trainX = subdataX[1]
                    trainY = subdataY[1]
                    #Test
                    testX = subdataX[0]
                    testY = subdataY[0]
                elif(i == 4):
                    #Split
                    subdataX = np.split(dataX,[subsetNum*i])
                    subdataY = np.split(dataY,[subsetNum*i])
                    #Train
                    trainX = subdataX[0]
                    trainY = subdataY[0]
                    #Test
                    testX = subdataX[1]
                    testY = subdataY[1]
                else:
                    #Split
                    subdataX = np.split(dataX,[subsetNum*i,subsetNum*(i+1)])
                    subdataY = np.split(dataY,[subsetNum*i,subsetNum*(i+1)])
                    #Train
                    trainX = np.concatenate((subdataX[0],subdataX[2]),axis=0)
                    trainY = np.concatenate((subdataY[0],subdataY[2]),axis=0)
                    #Test
                    testX = subdataX[1]
                    testY = subdataY[1]
                
				#Normalize train and test sets
                normalize(trainX,testX)
               
			   #Soft Margin SVM
                weight_vector_svm = train_classifier(trainX,trainY,learnRate,hinge_loss,lambdaVal,l2_reg)
                pred_y_svm = test_classifier(weight_vector_svm,testX)
                acc_svm = compute_accuracy(testY,pred_y_svm)
                acc_svm_avg+=acc_svm
                
				#Logistic Regression
                weight_vector_log = train_classifier(trainX,trainY,learnRate,logistic_loss)
                pred_y_log = test_classifier(weight_vector_log,testX)
                acc_log = compute_accuracy(testY,pred_y_log)
                acc_log_avg+=acc_log
            
			print("Softmargin SVM",',learn rate:',learnRate,',lambda:',lambdaVal)
            print("Accuracy",acc_svm_avg/5)
            print("Logistic Regression",",learn rate",learnRate)
            print("Accuracy",acc_log_avg/5)
            print()
    return None

if __name__ == "__main__":

    main()