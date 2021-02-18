'''Code to ML Non-palindromic form HS data using NN Regressors:
    a) ML embedding 'p' weights from low order coefficients of HS
    b) ML embedding 'p' weights from high order coefficients of HS
    To run: Set variable 'HS_low_check' = '1'/'0' to run investigation a/b respectively, run cells consecutively.
    ...uncomment matplotlib import and final plotting cell if wish to plot the training data
    ...there is further functionality associated to 'acc_list' (line 76) which can be uncommented with further code in training (lines 93-104) and outputs (lines 111-113) to give binned classification-style measures of learning
    '''

HS_low_check = 1 #...select '1' / '0' if wish to use lower/higher order coefficients respectively

#Import Libraries
import numpy as np
#import matplotlib.pyplot as plt
from math import floor
from ast import literal_eval
from tensorflow import keras

#%% #Import data of HS coefficients for orders: 0->100 (low) & 1000->1009 (high)
HS_low, HS_high, p_vecs = [], [], []
repeats_indices = []
#Import the HS coefficients
with open('../Data/Regressor-HS.txt','r') as HS_datafile: #...confirm correct file path
    linecounter = 0
    for line in HS_datafile:
        linecounter += 1
        coeffs = literal_eval(line.replace('{','[').replace('}',']'))
        while len(coeffs[0]) < 101:  #...pad coefficient vectors so all same length, if necessary
            coeffs[0].append(0)
        while len(coeffs[1]) < 10:
            coeffs[1].append(0)
        if coeffs[0] in HS_low and coeffs[1] in HS_high: #...record lines corresponding to repeated functions
            repeats_indices.append(linecounter)
            continue
        HS_low.append(coeffs[0])
        HS_high.append(coeffs[1])
#Import the vector of embedding weights
with open('../Data/Regressor-weights.txt','r') as p_vec_datafile: #...confirm correct file path
    linecounter = 0
    for line in p_vec_datafile:
        linecounter += 1
        if linecounter in repeats_indices: continue #...ignore lines asssociated to repeated functions
        p_vecs.append(literal_eval(line.replace('{','[').replace('}',']')))
del(HS_datafile,p_vec_datafile,coeffs,linecounter)

#Choose Data to ML
if HS_low_check: inputs = [[float(x) for x in y] for y in HS_low]  
else: inputs = [[float(x) for x in y] for y in HS_high]  
values = [[float(x) for x in y] for y in p_vecs]    
data_size = len(inputs)

#Number of k-fold cross-validations to perform
k = 5   #... k = 5 => 80(train) : 20(test) splits approx.
#Combine, shuffle, & split data (shuffle not really necessary as originally a random generation)
ML_data = [[inputs[index],values[index]] for index in range(data_size)]
np.random.shuffle(ML_data)
s = int(floor(data_size/k)) #number of HS in each validation split
Training_data, Training_values, Testing_data, Testing_values = [], [], [], []
for i in range(k):
    Training_data.append([HS[0] for HS in ML_data[:i*s]]+[HS[0] for HS in ML_data[(i+1)*s:]])
    Training_values.append([HS[1] for HS in ML_data[:i*s]]+[HS[1] for HS in ML_data[(i+1)*s:]])
    Testing_data.append([HS[0] for HS in ML_data[i*s:(i+1)*s]])
    Testing_values.append([HS[1] for HS in ML_data[i*s:(i+1)*s]])

del(ML_data)

#%% #Create and Train NN
#Define NN hyper-parameters
def act_fn(x): return keras.activations.relu(x,alpha=0.01) #...leaky-ReLU activation
number_of_epochs = 20           #...number of times to run training data through NN
size_of_batches = 32            #...number of datapoints the NN sees per iteration of optimiser (high batch means more accurate param updating, but less frequently) 
layer_sizes = [1024,1024,1024,1024]        #...number and size of the dense NN layers

#Define lists to record training history and learning measures
hist_data = []               #...training data (output of .fit(), used for plotting)
metric_loss_data = []        #...list of learning measure losses [MAE,MSE,logcosh] (could include huber also)
#acc_list = []                #...list of accuracy ranges [\pm ?, \pm ?, \pm ?] - tried %s, decided as many test values ~1 not as logical!

#Train k independent NNs for k-fold cross-validation (learning measures then averaged over)
for i in range(k):
    #Setup NN
    model = keras.Sequential()
    for layer_size in layer_sizes:
        model.add(keras.layers.Dense(layer_size, activation=act_fn))
        model.add(keras.layers.Dropout(0.05)) #...dropout layer to reduce chance of overfitting to training data
    model.add(keras.layers.Dense(3))
    model.compile(optimizer='adam', loss='logcosh') #...can switch to 'mse' loss if wish - results negligibly different 
    #Train NN
    hist_data.append(model.fit(Training_data[i], Training_values[i], batch_size=size_of_batches, epochs=number_of_epochs, shuffle=1, validation_split=0., verbose=1))
    #Test NN
    predictions = model.predict(Testing_data[i]) 
    metric_loss_data.append([np.mean(keras.losses.MAE(Testing_values[i],predictions)),np.mean(keras.losses.MSE(Testing_values[i],predictions)),np.mean(keras.losses.logcosh(Testing_values[i],predictions))])
    '''
    range1, range2, range3 = 0, 0, 0
    for test in range(len(predictions)):
        if np.mean(abs(Testing_values[i][test]-predictions[test])) < 1: 
            range1 += 1
            range2 += 1
            range3 += 1
        elif np.mean(abs(Testing_values[i][test]-predictions[test])) < 2: 
            range2 += 1
            range3 += 1
        elif np.mean(abs(Testing_values[i][test]-predictions[test])) < 3:
            range3 += 1
    acc_list.append([range1/len(predictions),range2/len(predictions),range3/len(predictions)])
    '''

#Output averaged learning measures
print('####################################')
print('Average measures:')
print('Loss data [MAE, MSE, LogCosh]:',[[np.mean([loss[0] for loss in metric_loss_data]),np.std([loss[0] for loss in metric_loss_data])/np.sqrt(k)],[np.mean([loss[1] for loss in metric_loss_data]),np.std([loss[1] for loss in metric_loss_data])/np.sqrt(k)],[np.mean([loss[2] for loss in metric_loss_data]),np.std([loss[2] for loss in metric_loss_data])/np.sqrt(k)]])
#print('Accuracy (within range1):',np.mean([acc[0] for acc in acc_list]),'\pm',np.std([acc[0] for acc in acc_list])/np.sqrt(k))
#print('Accuracy (within range2):',np.mean([acc[1] for acc in acc_list]),'\pm',np.std([acc[1] for acc in acc_list])/np.sqrt(k))
#print('Accuracy (within range3):',np.mean([acc[2] for acc in acc_list]),'\pm',np.std([acc[2] for acc in acc_list])/np.sqrt(k))

'''
#%% #Plot History Data
#Note history data is taken during training (not as good as test data statistics)
hist_label = range(1,len(hist_data)+1)
plt.close("all")
   
plt.figure('Loss History')
for i in range(len(hist_data)):
    plt.plot(range(1,number_of_epochs+1),hist_data[i].history['loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim(0,number_of_epochs+1)
plt.ylim(0)
plt.xticks(range(number_of_epochs+1))
plt.legend(hist_label, loc='upper right')
plt.grid()
plt.show()
'''