# Predicting Forest Cover Type

These are my solutions to the Kaggle challenge with the same name. The challenge
is to predict 1 out of 7 categories signifying the predominant kind of tree
cover in an area of 30x30 meter cells from 54 input variables. The training set
consists of 15120 data points and the test set is comprised of 565892
observations.

## forest_cover_type_prediction_using_a_neural_net.py

This python script can be run from the command line and expects to be in the
folder that contains the data. It produces a file called prediction.csv with the
predictions. It does that by first training a neural network with one hidden
layer on a training set with 80% of the observations (20% were reserved for the
cross-validation set) and then using this network to make predictions on the
test data set. The network architecure was copied from Blackard, Jock and Denis
(2000)[0], the original publication this dataset was used in. The model is
trained using Keras and the TensorFlow backend. Defaults for the fit procedure
are hard coded.

The prediction accuracy achieved is 70%, thereby matching the accuracy achieved
by Blackard et al. The prediction accuracy of the CV set was at around 84%,
which was very close to the training accuracy. This allows the conclusion that
the test set accuracy could be improved (through feature engineering).

[0] Blackard, Jock A. and Denis J. Dean. 2000. "Comparative Accuracies of
Artificial Neural Networks and Discriminant Analysis in Predicting Forest Cover
Types from Cartographic Variables." Computers and Electronics in Agriculture
24(3):131
