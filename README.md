# PCML Project 2
MATLAB code and report for [Project 2](http://icapeople.epfl.ch/mekhan/pcml15/project-2/objectDetection.html) of the [Pattern Recognition and Machine Learning course](http://icapeople.epfl.ch/mekhan/pcml15.html) at EPFL. The goal of the project was to classify images that belong to one of four categories: Horse, Airplane, Car, or Other.

1) Download [piotr_toolbox](http://cvlabwww.epfl.ch/~cjbecker/tmp/piotr_toolbox.zip) and place in `code`.

2) Download the training and test data from the [Project website](http://icapeople.epfl.ch/mekhan/pcml15/project-2/index.html). Place both `.mat` files in the `code/data` folder.

3) Run the following functions in the `code/pca` folder so the k-fold training and testing data for binary and multi-class prediction are already computed for each of the models. The number of folds and the number of principal components can be set on `Line 5` of both scripts.

* `precomputeVectors_binary.m`  (for classifying between horse/plane/car and others)
* `precomputeVectors.m`         (for classying between the four)

4) To apply the following models for classification:

* k-NN (k- nearest neighbors)
* Neural Network
* Random Forest
* SVM

run `test_<MODEL_NAME>.m` in the corresonding model's folder:

Note: let me know if any of the links do not work.


