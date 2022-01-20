#Import Libraries & relevant functions
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#%% #Import data
#Non-palindromic & Palindromic HS coefficients for orders: 0->9 (low) & 1000->1009 (high)
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
    
#Import datafiles of fake & real HS coefficients for orders 0->99
fake_HS, real_HS = [], []
with open('../Data/BC-fakeHS.txt','r') as fakeHSfile: #...confirm correct file path
    for line in fakeHSfile:
        fake_HS.append(literal_eval(line))
with open('../Data/BC-realHS.txt','r') as realHSfile: #...confirm correct file path
    for line in realHSfile:
        real_HS.append(literal_eval(line))
del(fakeHSfile,realHSfile,line)

#%% #Perform PCA
#Define the standardisers and PCA projectors
scaler_g = StandardScaler()
pca_G = PCA(n_components=2)
scaler_rf = StandardScaler()
pca_rf = PCA(n_components=2)

#Perform standardisation and PCA
#Gorenstein
scaler_g.fit(NonPalinHS_low+PalinHS_low)
transformed_NonPalin = scaler_g.transform(NonPalinHS_low)
transformed_Palin = scaler_g.transform(PalinHS_low)
pca_G.fit(np.concatenate((transformed_NonPalin,transformed_Palin)))
pca_nonpalin = pca_G.transform(transformed_NonPalin)
pca_palin = pca_G.transform(transformed_Palin)
#Real/Fake
scaler_rf.fit(fake_HS+real_HS)
transformed_fake = scaler_rf.transform(fake_HS)
transformed_real = scaler_rf.transform(real_HS)
pca_rf.fit(np.concatenate((transformed_fake,transformed_real)))
pca_fake = pca_rf.transform(transformed_fake)
pca_real = pca_rf.transform(transformed_real)

#%% #Plotting Grenstein 2d PCA
#Gorenstein
plt.figure('Gorenstein PCA')
plt.scatter(pca_palin[:,0],pca_palin[:,1],label='Gorenstein',alpha=0.1,color='blue')
plt.scatter(pca_nonpalin[:,0],pca_nonpalin[:,1],label='Non-Gorenstein',alpha=0.1,color='red')
#plt.xlabel('PCA component 1')
#plt.ylabel('PCA component 2')
leg=plt.legend(loc='upper right')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
#plt.grid()
#plt.savefig('./GorensteinPCA.pdf')

#%% #Plotting Real/Fake 2d PCA
#Real/Fake
plt.figure('Real/Fake PCA')
plt.scatter(pca_fake[:,0],pca_fake[:,1],label='Fake',alpha=0.3,color='blue')
plt.scatter(pca_real[:,0],pca_real[:,1],label='Real',alpha=0.3,color='red')
#plt.xlabel('PCA component 1')
#plt.ylabel('PCA component 2')
#plt.xlim(-5,100)
leg=plt.legend(loc='upper right')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
#plt.grid()
#plt.savefig('./RealFakePCA_2d.pdf')

#%% #Plotting Real/Fake 1d PCA
#Real/Fake
fig = plt.figure('Real/Fake PCA')
ax = fig.add_subplot(1, 1, 1)
plt.scatter(pca_fake[:,0],np.zeros(len(pca_fake)),label='Fake',alpha=0.3,color='blue')
plt.scatter(pca_real[:,0],np.zeros(len(pca_real)),label='Real',alpha=0.3,color='red')
leg=plt.legend(loc='upper right')
for lh in leg.legendHandles: 
    lh.set_alpha(1)
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.yticks([]) 
#plt.savefig('./RealFakePCA_1d.pdf',bbox_inches='tight')

