# HilbertSeriesML
Scripts to generate and analyse HS data.

Data-files are listed in the `Data` folder, they provide datasets of (fake) Hilbert Series (HS) coefficients, as well as datasets of equivalent HS function parameters used in the machine learning (ML) investigations.

Code-scripts are listed in the `Scripts` folder:

* `DataGeneration.nb` - 
  Mathematica notebook to generate the listed datasets in the Data folder (mathematica was an order of magnitude faster in Taylor expansion calculation than python's Sympy). 
  Run each section to generate the respective data for each investigation as described. 
  Note also that the BC-realHS.txt dataset is a list of HS coefficients for Fano 3-dimensional taken from the GRDB.
  
* `ML_Regressor.py` - 
  Python script to ML the embedding weights of fake-generated HS in refined form from their HS expansion coefficients of orders 0-100 or 1000-1009, as described in section III.B. 
  Set the 'HS_low_check' variable to '1'/'0' to use the lower or higher order coefficients repsectively for ML, then run the cells sequentially.
  
* `ML_Classifier.py` -
  Python script to ML the Gorenstein index or dimension of fake-generated HS in Palindromic-refined form from their HS expansion coefficients of orders 0-100 or 1000-1009, as described in section III.B.
  Set the 'Param_check' variable to '1'/'0' to ML Gorenstein index or dimension respectively; then set 'HS_low_check' variable to '1'/'0' to use the lower or higher order coefficients repsectively for ML, then run the cells sequentially.
  
* `ML_BinaryClassification.py` - 
  Python script to use ML to distinguish the refined/Palindromic-refined form of fake-generated HS, or whether the HS coefficients came from a fake-generated HS or from real HS from the GRDB, as described in sections III.C & III.D respectively. 
  Set the 'Palin_check' variable to '1'/'0' to perform the classification to distinguish HS function form or distinguishing fake from real investigations respectively; then set 'HS_low_check' variable to '1'/'0' to use the lower or higher order coefficients repsectively for ML, then run the cells sequentially.

* `ML_Complete_Intersection/` -
  Dicrectory contains python scripts to ML wether a fake HS corresponds to a complete intersection (CI) or not.
  As input `Data/CI_big.db` is used which contains lists of successive quotients of Taylor expansion coefficients.
  Further details about this investigation can be found in Section III.E in the paper.
  Please first unzip the compressed `Data/CI_big.db` file before running the scripts.
  Run `ml_ci_rf.py` for the principal component analysis + random forest experiment.
  Run `ml_ci_nn.py` for the principal component analysis + neural network experiment.
  A list of parameters at the beginning of each script facilitates the fine-tuning of the machine learning investigations.
  Please note the previous iteration of this subinvestigation has respective scripts provided in the PreviousWork subdirectory.

* `PCAs.py` -
  Python script to perform principal component analysis of the binary classification problems. Run the scripts first cell to import used libraries, then the second cell to import the data, ensuring correct filepaths. The latter cells then create and fit the scalers and pcas.

