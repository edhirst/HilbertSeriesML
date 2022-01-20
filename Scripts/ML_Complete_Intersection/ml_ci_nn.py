"""Machine learn complete intersections from (fake) Hilbert series (HS).

Script runs 10-fold cross validation studying if principal component
analysis + dense neural network can discriminate between fake HS that come
from complete and non-complete intersections.

Input: successive quotients h_i/h_{i+1} of the fake HS coefficients
       hs[i]=h_i for i=0,...,N_TERMS, i.e.,
       [hs[i]/hs[i+1] for i in range(N_TERMS)]

See list of parameters below to adjust training."""
import scipy.stats
import numpy as np

from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.math import confusion_matrix

from load_data import load_data

# pylint: disable=invalid-name

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
# N_TERMA = 1,...,301 (upper limit of saved terms in dataset)
N_TERMS = 301
# number of epochs to train NN
N_EPOCHS = 20
# batch sieze
B_SIZE = 32
# k fold validation
K_FOLD = 10
# ------------------------------------------------------------------------------

X, y = load_data(N_TERMS)

# run ML algorithms

print("\nLearning complete intersections from fake Hilbert series "
      "coefficients using PCA + dense NN.")
print("Learning on 10% of the data [h_i/h_{i+1}|"
      f"i=0,...,{len(X[0])}]\n")

accuracy_scores = []
mcc_scores = []
cm_scores = []


kf = KFold(n_splits=K_FOLD)
for train, test in kf.split(X):
    # train on the small chuncks
    test, train = train, test

    pca = PCA(n_components=0.99999)
    # fit PCA only on training data
    pca.fit(X[train], y[train])

    # project all data
    X = pca.transform(X)

    # Now we train an NN on the dimensionality reduced data
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=X[train].shape[1]))
    for _ in range(4):
        model.add(keras.layers.Dense(
            32, activation=keras.layers.LeakyReLU(alpha=0.01)))
        model.add(keras.layers.Dropout(rate=0.05))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    history = model.fit(X[train], y[train], epochs=N_EPOCHS, batch_size=B_SIZE,
                        shuffle=1, validation_split=0.2, verbose=0)
    fitness = model.evaluate(X[test], y[test])

    accuracy_scores.append(fitness[1])

    cm = confusion_matrix(y[test], tf.math.round(model.predict(X[test])))
    cm_scores.append(cm.numpy())

    y_pred = tf.math.round(model.predict(X[test])).numpy()
    mcc_scores.append(matthews_corrcoef(y[test], y_pred))

print("\n\nEnd training! ")

accuracy_std_deviation = np.std(accuracy_scores)
print("accuracy:")
print(f"mean: {np.mean(accuracy_scores)}")
print(f"standard deviation: {accuracy_std_deviation}")
# compute confidence interval
cfi = scipy.stats.t.interval(0.95,
                             len(accuracy_scores)-1,
                             loc=np.mean(accuracy_scores),
                             scale=scipy.stats.sem(accuracy_scores))
print(f"confidence interval of 95%: +/-{(cfi[1]-cfi[0])/2}\n")

print("mcc:")
print(f"mean: {np.mean(mcc_scores)}")
print(f"standard deviation: {np.std(mcc_scores)}\n")

print("cm:")
print(f"mean:\n{np.mean(cm_scores, axis=0)}\n")
print(f"standard deviation:\n{np.std(cm_scores, axis=0)}\n")
