# Reviews_Analysis
## CS 155 MiniProject 1 -- Kaggle

### Description of MiniProject

"In this competition, you are to use the training data (training_data.txt) to
come up with predictions for the test data (test_data.txt). The file
'training_data.txt' contains 20,000 reviews that you can use to train your
model. The first row is a header listing the label heading and 1,000 selected
words in the bag-of-words model. Each subsequent row contains a label
indicating the sentiment of that review (1-2 stars or 4-5 stars) followed by
the count of each word in the given Amazon review. The file 'test_data.txt'
contains a header and 10,000 reviews in the same format, but excluding the
label.""

### Division of Tasks

We will implement various models (neural networks, random forests, SVM, etc)
with different hyperparameters in separate functions. For each model, we will
use 5-fold cross validation to train with data from the training_data.text and
then make predictions on test_data.txt.
