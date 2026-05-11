import sys


from Legecy import utils
from Legecy import miniROCKET as mr
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

def normalize(x_train , x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled

def SVC_classifier(x_train , y_train, x_test, y_test,verbose = False,alpha= None):
    # Normalize the training data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

        
    classifier = SVC(kernel='linear')
    classifier.fit(x_train_scaled, y_train)
    pred_train = classifier.predict(x_train_scaled)
    train_accuracy = accuracy_score(y_train, pred_train)
    if verbose == True: 
        print('train: ',train_accuracy)
    # Make predictions on the normalized test data
    predictions = classifier.predict(x_test_scaled)
    
    accuracy = accuracy_score(y_test, predictions)
    return accuracy , (x_train_scaled,x_test_scaled) , classifier

def classic_classifier(x_train , y_train, x_test, y_test,balanced=True,verbose = False,alpha= None):
    # Normalize the training data
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Train the classifier
    if type(None) == type(alpha):
        if balanced == True: 
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),class_weight = 'balanced' )
        else:
            classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10) )
    else: 
        if balanced == True: 
            classifier = RidgeClassifierCV(alphas=alpha,class_weight = 'balanced')
        else:
            classifier = RidgeClassifierCV(alphas=alpha)
    classifier.fit(x_train_scaled, y_train)
    pred_train = classifier.predict(x_train_scaled)
    train_accuracy = accuracy_score(y_train, pred_train)
    if verbose == True: 
        print('train: ',train_accuracy)
    # Make predictions on the normalized test data
    predictions = classifier.predict(x_test_scaled)
    
    accuracy = accuracy_score(y_test, predictions)
    return accuracy , (x_train_scaled,x_test_scaled) , classifier
    

def vector_creator(index = 0 ):
    indices = np.array((
        0,1,2,0,1,3,0,1,4,0,1,5,0,1,6,0,1,7,0,1,8,
        0,2,3,0,2,4,0,2,5,0,2,6,0,2,7,0,2,8,0,3,4,
        0,3,5,0,3,6,0,3,7,0,3,8,0,4,5,0,4,6,0,4,7,
        0,4,8,0,5,6,0,5,7,0,5,8,0,6,7,0,6,8,0,7,8,
        1,2,3,1,2,4,1,2,5,1,2,6,1,2,7,1,2,8,1,3,4,
        1,3,5,1,3,6,1,3,7,1,3,8,1,4,5,1,4,6,1,4,7,
        1,4,8,1,5,6,1,5,7,1,5,8,1,6,7,1,6,8,1,7,8,
        2,3,4,2,3,5,2,3,6,2,3,7,2,3,8,2,4,5,2,4,6,
        2,4,7,2,4,8,2,5,6,2,5,7,2,5,8,2,6,7,2,6,8,
        2,7,8,3,4,5,3,4,6,3,4,7,3,4,8,3,5,6,3,5,7,
        3,5,8,3,6,7,3,6,8,3,7,8,4,5,6,4,5,7,4,5,8,
        4,6,7,4,6,8,4,7,8,5,6,7,5,6,8,5,7,8,6,7,8
    ), dtype = np.int32).reshape(84, 3)
    vector = np.full(9 , -1)
    vector[indices[index]] = 2
    return vector

def dummy_transformer(x_train, x_test, parameter):
    """
        designed like original miniROCKET but designed and coded by Alireza
    """
    
    length = x_train.shape[1]
    
    dilation, n_dilation, biases = parameter 
    n_param = len(biases)
    train_trans = np.zeros((x_train.shape[0],n_param))
    test_trans = np.zeros((x_test.shape[0],n_param))
    print(dilation)
    
    
    for p in range(len(biases)):
        v = vector_creator(p%84)
        D = dilation[p//84]
        v = utils.dilute(v, D)
        B = biases[p]
        C = mr.create_rter(np.array([B]),length=length)
        for i in range(x_train.shape[0]):
            conv = np.convolve(x_train[i],v,'same')  
            conv = conv[:length]
            train_trans[i,p] = mr._PPV(conv,C).mean()
        for i in range(x_test.shape[0]):
            conv = np.convolve(x_test[i],v,'same')  
            conv = conv[:length]
            test_trans[i,p] = mr._PPV(conv,C).mean()
        
    return train_trans, test_trans

def mcmc_transformer(x_train, x_test, parameter):
    length = x_train.shape[1]
    n_param = len(parameter)
    train_trans = np.zeros((x_train.shape[0],n_param))
    test_trans = np.zeros((x_test.shape[0],n_param))
    
    for p in range(n_param):
        v_index , D, S , B =parameter[p]
        v =  vector_creator(v_index)
        v = utils.dilute(v, D)
        C = mr.create_rter(B,length=length)
        for i in range(x_train.shape[0]):
            conv = np.convolve(x_train[i],v,'same')  
            conv = conv[:length]
            train_trans[i,p] = mr._PPV(conv,C).mean()
        for i in range(x_test.shape[0]):
            conv = np.convolve(x_test[i],v,'same')  
            conv = conv[:length]
            test_trans[i,p] = mr._PPV(conv,C).mean()
            
    return train_trans, test_trans