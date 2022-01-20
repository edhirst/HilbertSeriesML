"""Machine learn complete intersections from (fake) Hilbert series (HS).

Script runs 10-fold cross validation studying if principal component
analysis + random forest can discriminate between fake HS that come from
complete and non-complete intersections.

Input: successive quotients h_i/h_{i+1} of the fake HS coefficients
hs[i]=h_i for i=0,...,N_TERMS, i.e., [hs[i]/hs[i+1] for i in range(N_TERMS)]

Change the number of successive quotients by adjusting the parameter below."""
from ml_rf_pipeline import run_pipeline_cross_val
from load_data import load_data

## pylint: disable=invalid-name


# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
# N_TERMS = 0,...,301 (upper limit of saved terms in dataset)
N_TERMS = 301
# ------------------------------------------------------------------------------

X, y = load_data(N_TERMS)

# run ML algorithms

print("\nLearning complete intersections from fake Hilbert series "
      "coefficients using PCA + random forest.")
print("Learning on 10% of the data [h_i/h_{i+1}|"
      f"i=0,...,{len(X[0])}]\n")

run_pipeline_cross_val(X,
                       y,
                       10,
                       invert_folds=True,
                       log_normalise=False,
                       use_pca=True)
