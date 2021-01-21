# SentimentAnalysis
Sentiment Analysis written in Python for restaurant review data.

# About
This is an implementation of a Logistic Regression model for conducting Sentiment Analysis on natural language data. The data is restaurant review data, and the Logistic Regression was implemented from scratch.

# Metrics on Validation Set

### Model output for: Naive Bayes - bag-of-words baseline

Precision: 1.0

Recall: 0.5

F1 metric: 0.667

### Model output for: Logistic Regression classifier

Precision: 1.0

Recall: 1.0

F1 Score: 1.0

# Metrics on Held Out Set using Logistic Regression

F1 Score: 0.943

# Experiments

  I first tried to experiment with Gensim. After realizing that using word embeddings would require
  some more research, I instead created my own features. The 'feature' module contains all of the functions
  that extract features from the data, and they are loaded in by a generator. The raw features resulted in
  a snowball effect on the weights, and would cause an overflow in the sigmoid function. To combat this,
  I implemented a min-max scaler, which gets fitted to the training data and scales both the training
  and testing data, and now the weights stay generally between +/- 10.0

  I wanted to see which features were most important for the classification, so I wrote a method for the Logistic
  Regression class which returns a list of tuples. Each tuple looks like (<feature-name>, <associated-weight>),
  which essentially shows the relative importance of the features. The resulting list is sorted from most to least
  important. We can see that the output of this function as follows:

		Feature: count_positive_words_in_list ---> Weight: 10.621614364494112
		Feature: count_negative_words_in_list ---> Weight: -10.352998966086748
		Feature: count_words_in_data ---> Weight: 1.870792312356487
		Feature: count_numbers ---> Weight: -1.5982889214490417
		Feature: count_exclamation_mark ---> Weight: 0.8259427767424262
		Feature: bias_term ---> Weight: -0.44724295456039054
		Feature: count_uppercase_word ---> Weight: 0.4325877129417038



  Additionally, I wanted to tune the learning rate hyper-parameters. I randomly sampled 100 logistic regression models
  with varying learning rates. I trained all of these models on the same training data and evaluated them all on the
  testing data. Finally, I printed out a list of tuples of the form (learning-rate, f1-score) sorted by the F1
  score. I found that there was a general preference for lower learning rates, which was already known. To see the
  graph of this, run this module with show_graph parameter set to True in the main() function like this:
                  'main(show_graph=True)'
  If I had some more time, I would have run the the following tests:

	  1) Testing learning rates sampled exponentially from the interval [10e-7, 0.5], perhaps using more than 100 samples.
	  2) Testing learning rate diminishing functions, ie. linear decay, exponential decay, etc.
	  3) Properly implementing batch-training and tuning the batch-size parameter.
	  4) Created more informative graphs for the current experiment.

  Finally, I ran a preprocessed the data by turning it all to lowercase. If I had extra time, I would have finished
  the implementation of add_negation.
