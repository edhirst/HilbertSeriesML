'''Code to ML Palindromic form HS data using NN Classifiers:
    a) ML Gorenstein index
    b) ML HS dimension
    ...for...
    1) low order coefficients of HS
    2) high order coefficients of HS (log-normalised)
    To run: Set variable 'Param_check' = '1'/'0' to ML Gorenstein/Dimension then 'HS_low_check' = '1'/'0' to use low/high order coefficients in line with investigations a/b with 1/2 respectively, run cells consecutively.
    ...uncomment matplotlib import and final plotting cell if wish to plot the training data
    '''

Param_check = 1  #...select '1' / '0' if wish to ML Gorenstein-index / dimension respectively (investigations a / b)
HS_low_check = 1 #...select '1' / '0' if wish to use lower/higher order coefficients respectively (investigations 1 / 2)

#Import Libraries
import numpy as np
#import matplotlib.pyplot as plt
from math import floor
from ast import literal_eval
from copy import deepcopy
from tensorflow import keras

#%% #Import data from mathematica for Palin-HS-> r or d
HS_low, HS_high, Gorenstein, Dimension = [], [], [], []
repeats_indices = []
with open('../Data/Classifier-HS.txt','r') as HS_datafile: #...confirm correct file path
    linecounter = 0    
    for line in HS_datafile:
        linecounter += 1
        coeffs = literal_eval(line.replace('{','[').replace('}',']'))
        while len(coeffs[0]) < 101:  #...padding if necessary
            coeffs[0].append(0)
        while len(coeffs[1]) < 10:
            coeffs[1].append(0)
        if coeffs[0] in HS_low and coeffs[1] in HS_high: #...record lines corresponding to repeated functions
            repeats_indices.append(linecounter)
            continue
        HS_low.append(coeffs[0])
        HS_high.append(coeffs[1])
        
with open('../Data/Classifier-params.txt','r') as HSfnparam_datafile: #...confirm correct file path
    linecounter = 0 
    for line in HSfnparam_datafile:
        linecounter += 1
        if linecounter in repeats_indices: continue #...ignore lines asssociated to repeated functions
        coeffs = literal_eval(line.replace('{','[').replace('}',']'))
        Gorenstein.append(coeffs[1])
        Dimension.append(coeffs[2])
del(HS_datafile,HSfnparam_datafile,coeffs,linecounter)

#%% Make Palin ML data
#Select which investigation to perform
if HS_low_check:    inputs = [[float(x) for x in y][1:] for y in HS_low]   #...note will truncate vectors to same length
else:               inputs = [[float(x) for x in y][1:] for y in HS_high]
if Param_check: values = [float(x-1) for x in Gorenstein]   #...reinitiate values to start at 0 (for NN labelling-processing purposes)
else:           values = [float(x-1) for x in Dimension]
number_classes = int(max(values))+1
data_size = len(inputs)

#Remove HS with zeros in coefficient list (so can apply logarithm), then log-normalise HS coeffs
if not HS_low_check: #...only log-normalise the higher order coefficients
    inputs_old = deepcopy(inputs)
    values_old = deepcopy(values)
    inputs, values = [], []
    for index_1 in range(data_size):
        if not np.all(inputs_old[index_1]): continue #...if there's a zero entry in expansion => skip
        inputs.append([float(x) for x in np.log(inputs_old[index_1])]) #...remake into standard float 32 (so same as values for NN to process)
        values.append(values_old[index_1])
    data_size = len(inputs) #...reset to new value

ML_data = [[inputs[index],values[index]] for index in range(data_size)]
#Number of k-fold cross-validations to perform
k = 5   #... k = 5 => 80(train) : 20(test) splits approx.
#Shuffle data ordering (not really necessary as originally a random generation)
np.random.shuffle(ML_data)
s = int(floor(data_size/k)) #number of HS in each validation split
Training_data, Training_labels, Testing_data, Testing_labels = [], [], [], []
for i in range(k):
    Training_data.append([HS[0] for HS in ML_data[:i*s]]+[HS[0] for HS in ML_data[(i+1)*s:]])
    Training_labels.append([HS[1] for HS in ML_data[:i*s]]+[HS[1] for HS in ML_data[(i+1)*s:]])
    Testing_data.append([HS[0] for HS in ML_data[i*s:(i+1)*s]])
    Testing_labels.append([HS[1] for HS in ML_data[i*s:(i+1)*s]])

del(ML_data)

#%% #Create and Train NN
#Define NN hyper-parameters
def act_fn(x): return keras.activations.relu(x,alpha=0.01) #...leaky-ReLU activation
number_of_epochs = 20           #...number of times to run training data through NN
size_of_batches = 32            #...number of datapoints the NN sees per iteration of optimiser (high batch means more accurate param updating, but less frequently) 
layer_sizes = [1024,1024,1024,1024]        #...number and size of the dense NN layers

#Define lists to record training history and learning measures
hist_data = []               #...training data (output of .fit(), used for plotting)
cm_list, acc_list, mcc_list = [], [], []    #...lists of measures

#Train k independent NNs for k-fold cross-validation (learning measures then averaged over)
for i in range(k):
    #Setup NN
    model = keras.Sequential()
    for layer_size in layer_sizes:
        model.add(keras.layers.Dense(layer_size, activation=act_fn))
        #model.add(keras.layers.Dropout(0.2)) #...dropout layer to reduce chance of overfitting to training data
    model.add(keras.layers.Dense(number_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    #Train NN
    hist_data.append(model.fit(Training_data[i], Training_labels[i], batch_size=size_of_batches, epochs=number_of_epochs, shuffle=1, validation_split=0., verbose=1))
    #Test NN - Calculate learning measures: confusion matrix, accuracy, MCC
    cm = np.zeros(shape=(number_classes,number_classes)) #initialise confusion matrix
    predictions = np.argmax(model.predict(Testing_data[i]),axis=1)
    for check_index in range(len(Testing_labels[i])):
        cm[int(Testing_labels[i][check_index]),int(predictions[check_index])] += 1      #...row is actual class, column is predicted class
    print()
    print('Normalised Confusion Matrix:') 
    print(cm/len(Testing_labels[i])) 
    print('Row => True class, Col => Predicted class')
    print()
    acc_list.append(sum([cm[n][n] for n in range(number_classes)])/len(Testing_labels[i])) #...sum of cm diagonal entries gives accuracy
    mcc_denom = np.sqrt((np.sum([np.sum([np.sum([cm[m,n] for n in range(number_classes)]) for m in list(range(number_classes)[:k])+list(range(number_classes)[k+1:])])*np.sum([cm[k,l] for l in range(number_classes)]) for k in range(number_classes)])) * (np.sum([np.sum([np.sum([cm[n,m] for n in range(number_classes)]) for m in list(range(number_classes)[:k])+list(range(number_classes)[k+1:])])*np.sum([cm[l,k] for l in range(number_classes)]) for k in range(number_classes)])))
    if mcc_denom != 0:
        mcc_list.append(np.sum([np.sum([np.sum([cm[k,k]*cm[l,m]-cm[k,l]*cm[m,k] for k in range(number_classes)]) for l in range(number_classes)]) for m in range(number_classes)]) / mcc_denom)
    else: 
        mcc_list.append(np.nan)

#Output averaged learning measures
print('####################################')
print('Average measures:')
print('Accuracy:',sum(acc_list)/k,'\pm',np.std(acc_list)/np.sqrt(k))
print('MCC:',sum(mcc_list)/k,'\pm',np.std(mcc_list)/np.sqrt(k))

'''
#%% #Plot History Data
#Note history data is taken during training (not as good as test data statistics)
hist_label = range(1,len(hist_data)+1)
plt.close("all")

plt.figure('Accuracy History')
for i in range(len(hist_data)):
    plt.plot(range(1,number_of_epochs+1),hist_data[i].history['accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xlim(0,number_of_epochs+1)
plt.ylim(0,1)
plt.xticks(range(number_of_epochs+1))
plt.yticks(np.arange(0,1.1,0.1))
plt.legend(hist_label, loc='upper left')
plt.grid()
plt.show()
   
plt.figure('Loss History')
for i in range(len(hist_data)):
    plt.plot(range(1,number_of_epochs+1),hist_data[i].history['loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xlim(0,number_of_epochs+1)
plt.ylim(0)
plt.xticks(range(number_of_epochs+1))
#plt.yticks(np.arange(0,1.3,0.1))
plt.legend(hist_label, loc='lower left')
plt.grid()
plt.show()
'''