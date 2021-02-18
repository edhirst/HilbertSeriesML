'''Code to binary classification ML problems: 
    a) distinguish HS forms (Non-palindromic form or Palindromic form)
    b) distinguish fake-HS from real-HS (artificailly generated or drawn from 3d Fano GRDB)
    To run: Set 'Palin_check' = '1'/'0' to run investigation a/b respectively, if running a set 'HScoeff_low_check' = '1'/'0' to ML lower/higher HS coefficients respectively, run cells consecutively.
    ...uncomment matplotlib import and final plotting cell if wish to plot the training data.'''

Palin_check = 1         #...set to '1' / '0' to perform Palindromic-NonPalindromic / Real-Fake investigation respecitvely
HScoeff_low_check = 0   #...set to '1' / '0' if wish to use lower/higher order coefficients respectively (only relevant for investigation a)

#Import Libraries& relevant functions
import numpy as np
#import matplotlib.pyplot as plt
from math import floor
from ast import literal_eval
from statistics import stdev
from tensorflow import keras

#%% #Import datafiles of Non-palindromic & Palindromic HS coefficients for orders: 0->9 (low) & 1000->1009 (high)
if Palin_check:
    #Select which coefficient order to ML
    PalinHS_low, PalinHS_high, NonPalinHS_low, NonPalinHS_high = [], [], [], []
    with open('../Data/BC-PalindromicHS.txt','r') as PalinData: #...confirm correct file path, used Palindromic data has 6 repeats of 10,000 HS generated (i.e. dataset ~ balanced)
        linecounter = 0    
        for line in PalinData:
            linecounter += 1
            coeffs = literal_eval(line.replace('{','[').replace('}',']'))
            while len(coeffs[0]) < 101:  #...padding if necessary
                coeffs[0].append(0)
            while len(coeffs[1]) < 10:
                coeffs[1].append(0)
            if coeffs[0] in PalinHS_low and coeffs[1] in PalinHS_high: #...skip lines corresponding to repeated functions
                continue
            PalinHS_low.append(coeffs[0])
            PalinHS_high.append(coeffs[1])

    with open('../Data/BC-NonPalindromicHS.txt','r') as NonPalinData: #...confirm correct file path
        linecounter = 0    
        for line in NonPalinData:
            linecounter += 1
            coeffs = literal_eval(line.replace('{','[').replace('}',']'))
            while len(coeffs[0]) < 101:  #...padding if necessary
                coeffs[0].append(0)
            while len(coeffs[1]) < 10:
                coeffs[1].append(0)
            if coeffs[0] in NonPalinHS_low and coeffs[1] in NonPalinHS_high: #...skip lines corresponding to repeated functions
                continue
            NonPalinHS_low.append(coeffs[0])
            NonPalinHS_high.append(coeffs[1])
    del(PalinData,NonPalinData,line,linecounter,coeffs)

    #Create Palin vs NonPalin ML Data
    #Number of k-fold cross-validations to perform
    k = 5   #... k = 5 => 80(train) : 20(test) splits approx.

    #Select which coefficients to ML
    if HScoeff_low_check:
        Palin = PalinHS_low
        NonPalin = NonPalinHS_low
    else:
        Palin = PalinHS_high
        NonPalin = NonPalinHS_high

    #Combine, shuffle, & split data 
    data_size = len(Palin)+len(NonPalin)
    ML_data = [[[float(HS_value) for HS_value in HS],0] for HS in NonPalin]+[[[float(HS_value) for HS_value in HS],1] for HS in Palin] #...make into floats for NN to process
    #ML_data = [[[np.log(float(HS_value)) for HS_value in HS],0] for HS in NonPalin]+[[[np.log(float(HS_value)) for HS_value in HS],1] for HS in Palin] #...make into floats for NN to process & log-normalise
    np.random.shuffle(ML_data)
    s = int(floor(data_size/k)) #number of HS in each validation split
    Training_data, Training_labels, Testing_data, Testing_labels = [], [], [], []
    for i in range(k):
        Training_data.append([HS[0] for HS in ML_data[:i*s]]+[HS[0] for HS in ML_data[(i+1)*s:]])
        Training_labels.append([HS[1] for HS in ML_data[:i*s]]+[HS[1] for HS in ML_data[(i+1)*s:]])
        Testing_data.append([HS[0] for HS in ML_data[i*s:(i+1)*s]])
        Testing_labels.append([HS[1] for HS in ML_data[i*s:(i+1)*s]])
        
    del(ML_data)
    
#%% #Import datafiles of fake & real HS coefficients for orders 0->99
else:
    fake_HS, real_HS = [], []
    with open('../Data/BC-fakeHS.txt','r') as fakeHSfile: #...confirm correct file path
        for line in fakeHSfile:
            fake_HS.append(literal_eval(line))
    with open('../Data/BC-realHS.txt','r') as realHSfile: #...confirm correct file path
        for line in realHSfile:
            real_HS.append(literal_eval(line))
    del(fakeHSfile,realHSfile,line)

    #Create ML Data
    #Number of k-fold cross-validations to perform
    k = 5   #... k = 5 => 80(train) : 20(test) splits approx.

    #Label, combine, shuffle, & split data into cross-validation sets
    data_size = len(fake_HS)+len(real_HS)
    ML_data = [[[float(HS_value) for HS_value in HS],0] for HS in fake_HS]+[[[float(HS_value) for HS_value in HS],1] for HS in real_HS] #...make into floats for NN to process
    #ML_data = [[[np.log(float(HS_value)) for HS_value in HS],0] for HS in fake_HS]+[[[np.log(float(HS_value)) for HS_value in HS],1] for HS in real_HS] #...make into floats for NN to process, and log-normalise so loss fn can handle also
    np.random.shuffle(ML_data)
    s = int(floor(data_size/k)) #number of HS in each validation split (last split may have more to account for discretising error)
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
cm_list = []                 #...confusion matrices (used for evaluation)
acc_list, mcc_list = [], []  #...learning measures

#Train k independent NNs for k-fold cross-validation (learning measures then averaged over)
for i in range(k):
    #Setup NN
    model = keras.Sequential()
    for layer_size in layer_sizes:
        model.add(keras.layers.Dense(layer_size, activation=act_fn))
        #model.add(keras.layers.Dropout(0.2)) #...dropout layer to reduce chance of overfitting to training data
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
    #Train NN
    hist_data.append(model.fit(Training_data[i], Training_labels[i], batch_size=size_of_batches, epochs=number_of_epochs, shuffle=1, validation_split=0., verbose=1))
    #Calculate learning measures: confusion matrix, accuracy, MCC
    cm = np.zeros(shape=(2,2)) #initialise confusion matrix
    predictions = [floor(x+0.5) for x in model.predict(Testing_data[i])]
    for check_index in range(len(Testing_labels[i])):
        cm[Testing_labels[i][check_index],predictions[check_index]] += 1      #...row is actual class, column is predicted class
    print()
    print('Normalised Confusion Matrix:') 
    print(cm/len(Testing_labels[i])) 
    print('Row => True class, Col => Predicted class')
    print('Class labels:',[0,1],'-->',['NonPalin/Fake','Palin/Real'])
    print()
    acc_list.append((cm[0][0]+cm[1][1])/len(Testing_labels[i])) #...sum of cm diagonal entries gives accuracy
    mcc_denom = np.sqrt((cm[1][1]+cm[0][1])*(cm[1][1]+cm[1][0])*(cm[0][0]+cm[0][1])*(cm[0][0]+cm[1][0]))
    if mcc_denom != 0:
        mcc_list.append((cm[1][1]*cm[0][0]-cm[0][1]*cm[1][0]) / mcc_denom)
    else: 
        mcc_list.append(np.nan)


#Output averaged learning measures
print('####################################')
print('Average measures:')
print('Accuracy:',sum(acc_list)/k,'\pm',stdev(acc_list)/np.sqrt(k))
print('MCC:',sum(mcc_list)/k,'\pm',stdev(mcc_list)/np.sqrt(k))

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
plt.legend(hist_label, loc='lower right')
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
plt.legend(hist_label, loc='upper right')
plt.grid()
plt.show()
'''
