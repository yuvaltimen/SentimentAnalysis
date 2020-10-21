import numpy as np
import sys
from math import log
from tqdm import tqdm
from helper_functions import sigmoid, feature_iterator, cross_entropy_loss, \
    generate_tuples_from_file, MinMaxScaler, evaluate_model, preprocess
import matplotlib.pyplot as plt
import random as rand

"""
Yuval Timen
Sentiment Analysis using Naive Bayes Classifier and Logistic Regression
"""


"""
Naive Bayes classifier for Sentiment Analysis
Uses bag-of-words as features
Uses add-1 smoothing
Optional: set use_binary=True to use Bernoilli Naive Bayes
"""
class SentimentAnalysis:
    
    def __init__(self, use_binary=False):

        self.priors = {'0': 0, '1': 0}
        self.positives_word_counts = {}
        self.negatives_word_counts = {}
        self.positives_denominator = 0
        self.negatives_denominator = 0
        self.vocabulary = set()
        self.binary = use_binary
        self.vocab_size = 0
    
    
    def train(self, examples):
        """Train our SentimentAnalysis model by updating the values of our counts
        Parameters:
            examples (list of tuples) - The list of featurized training examples: (ID, Data, Class)
        Returns:
            None
        """
        # First, we build our vocabulary
        # We can also update the priors here
        for tup in examples:
            self.vocabulary = self.vocabulary.union(set(tup[1].split()))
            if bool(int(tup[2])):
                self.priors['1'] += 1
            else:
                self.priors['0'] += 1
        # Normalize priors to be probabilities
        self.priors = {k: v/len(examples) for k, v in self.priors.items()}
        self.vocab_size = len(self.vocabulary)
        print(f"Size of vocabulary: |v| = {self.vocab_size}")
        
        # We split the examples into positives and negatives
        # We group them by their classification
        positive_examples = list(filter(lambda tup: bool(int(tup[2])), examples))
        negative_examples = list(filter(lambda tup: not bool(int(tup[2])), examples))
        
        # If we're using Binary-NB, let's remove all duplicate words in each document
        if self.binary:
            positives_document = [" ".join(set(tup[1].split())) for tup in positive_examples]
            negatives_document = [" ".join(set(tup[1].split())) for tup in negative_examples]
        else:
            positives_document = [tup[1] for tup in positive_examples]
            negatives_document = [tup[1] for tup in negative_examples]
        
        # Concatenate all the documents in each class
        positives_document = " ".join(positives_document)
        negatives_document = " ".join(negatives_document)
        
        # Update the word counts for all words in the positive document
        for w in positives_document.split():
            if w in self.positives_word_counts.keys():
                self.positives_word_counts[w] += 1
            else:
                self.positives_word_counts[w] = 1
        
        self.positives_denominator = sum([val for val in self.positives_word_counts.values()])
        
        # Update the word counts for all words in the negative document
        for w in negatives_document.split():
            if w in self.negatives_word_counts.keys():
                self.negatives_word_counts[w] += 1
            else:
                self.negatives_word_counts[w] = 1

        self.negatives_denominator = sum([val for val in self.negatives_word_counts.values()])
    
    
    def score(self, data):
        """Uses the models weights to calculate a score for the given data point
        Parameters:
            data (str) - The raw text data, ie. the middle element of the training tuple
        Returns:
            dict of values assigning to each class a probability
        """
        pos_score = log(self.priors['1'])
        neg_score = log(self.priors['0'])
        
        for w in data.split():
            if w in self.vocabulary:
                # Update score for positives
                if w in self.positives_word_counts.keys():
                    pos_score += log((self.positives_word_counts[w] + 1) / (self.positives_denominator + self.vocab_size))
                else:
                    pos_score += log(1 / (self.positives_denominator + self.vocab_size))
                # Update score for negatives
                if w in self.negatives_word_counts.keys():
                    neg_score += log((self.negatives_word_counts[w] + 1) / (self.negatives_denominator + self.vocab_size))
                else:
                    neg_score += log(1 / (self.negatives_denominator + self.vocab_size))
            else:
                continue
        
        return {'0': np.exp(neg_score), '1': np.exp(pos_score)}
                

    def classify(self, data):
        """Returns the argmax of the scores assigned to each class
        Parameters:
            data (str) - The raw text data, ie. the middle element of the training tuple
        Returns:
            str label, either '0' or '1'
        """
        score = self.score(data)
        if score['0'] >= score['1']:
            return '0'
        else:
            return '1'
    
    
    def featurize(self, data):
        """Featurizes the data by linking each feature to its corresponding value
        Parameters:
            data (str) - The raw text data, ie. the middle element of the training tuple
        Returns:
            list of tuples linking each feature to a value"""
        output = []
        for w in data.split():
            output.append((w, True))
        return output
        
        
    def describe_experiments(self):
        s = "No experiments ran with Naive Bayes classifier"
        return s
    
    
    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"


"""
Logistic Regression classifier for Sentiment Analysis
"""
class SentimentAnalysisImproved:

    def __init__(self, learning_rate=0.01, epsilon=0.0001):
        """Initialize the Logistic Regression class"""
        self.learning_rate = learning_rate
        self.weights = None  # Save this namespace
        self.epsilon = epsilon
        self.scaler = MinMaxScaler()


    def train(self, examples):
        """Train our Logistic Regression model by updating the weights in each iteration
        Uses stochastic gradient descent and the cross-entropy loss function to train
        Each iteration trains on a random subset of size batch_size from the training data
        Parameters:
            examples (list of tuples) - The list of tuples to train on
        Returns:
            None
        """
        x = self.featurize(examples)
        y = [float(tup[2]) for tup in examples]
        gradient_mag = float('inf')
        count_iters = 0
        loss_list = []
        gradient_mags = []
        num_epochs = 4000
        batch_size = 10
        
        self.weights = np.random.randn(x[0].shape[0]).astype('float128')
        print(f"weights: {self.weights}")
        
        while gradient_mag > self.epsilon:
            for i in rand.sample(range(len(y)), batch_size):
                count_iters += 1
                y_i = y[i]
                z = np.dot(x[i], self.weights)
                sig = sigmoid(z)
                loss_list.append(cross_entropy_loss(y_i, sig))
                gradient = (sig - y_i) * x[i]
                gradient_mag = np.sum(np.abs(gradient))
                gradient_mags.append(gradient_mag)
                
                self.weights = self.weights - (self.learning_rate * gradient)
            
        print(f"Done training! Total {count_iters} iterations")
        print(f"Final weights: {self.weights}")
        
        plt.plot(loss_list)
        plt.xlabel("Iteration #")
        plt.ylabel("gradient magnitude")
        plt.show()


    def score(self, data):
        """Uses the model's weights to calculate a score for the given data point
        Parameters:
            data (str) - The raw text data, ie. the middle element in the tuple
        Returns:
            dict - a mapping from each class to the associated score
        """
        X = self.featurize_one(data)
        X = self.scaler.scale(X)
        z = np.dot(X, self.weights)
        return sigmoid(z)


    def classify(self, data):
        """Returns the predicted class of the given data point
        Parameters:
            data (str) - The raw text data, ie. the middle element in the tuple
        Returns:
            str - label, either '0' or '1'
        """
        if self.score(data) < 0.5:
            return '0'
        else:
            return '1'


    def featurize_one(self, data):
        """Featurizes the data by returning a vector representation of the data
        Parameters:
            data (str) - The raw text data, ie. the middle element in the tuple
        Returns:
            np.array - the vector representation of the featurized data
        """
        ans = []
        for ft in feature_iterator():
            ans.append(ft(data))
            
        return np.array(ans)
    
    
    def featurize(self, all_examples):
        """Featurizes all the examples by returning a vector of vectors
        Parameters:
            all_examples (list of tuples) - All the training examples, each of the form (ID, data, label)
        Returns:
            np.array of np.arrays - The vectorized features
        """
        ans = []
        for example in all_examples:
            ans.append(self.featurize_one(example[1]))
        
        self.scaler.fit(ans)
        
        return [self.scaler.scale(vect) for vect in ans]
    
    
    # DISABLED:
    # I tried to implement Dense Word Embeddings using Gensim's word2vec
    # but I ran into issues of vector representations: since we don't know the
    # length of the testing or training example being fed in to the model, we
    # can't build a consistent vector embedding for it without sacrificing
    # information about word order. word2vec trains on the input data and then
    # returns the word embedding for each word in the input, but we want to
    # encode entire sentences. In order to keep the vectors the same length,
    # we would have to either sum or average all of the word embeddings in the
    # input, which would give us a bag-of-words embedding. Not what we want :(
    
    # def featurize(self, all_examples):
    #     """Use gensim's word2vec vectorizer to create dense embeddings for the input data
    #     Parameters:
    #         all_examples (list of tuples) - all training examples
    #     Returns:
    #         np.array - The vector embedding
    #     """
    #     all_sentences = [ex[1] for ex in all_examples]
    #     vectorizer = Word2Vec(min_count=2,
    #                           window=4,
    #                           size=300,
    #                           alpha=0.02
    #                           )
    #     vectorizer.build_vocab(all_sentences)
    #     vectorizer.train(all_sentences,
    #                      total_examples=vectorizer.corpus_count,
    #                      epochs=30)
    #     vectorizer.init_sims(replace=True)
    #     all_vectors = [[sum(vectorizer.wv[word])for word in sentence.split()] for sentence in all_sentences]
    #     return all_vectors
        

    def __str__(self):
        return "Logistic Regression classifier"


    def describe_experiments(self):
        s = """
    I first tried to experiment with Gensim. After realizing that using word embeddings would require
    some more research, I instead created my own features. The 'feature' module contains all of the functions
    that extract features from the data, and they are loaded in by a generator. The raw features resulted in
    a snowball effect on the weights, and would cause an overflow in the sigmoid function. To combat this,
    I implemented a min-max scaler, which gets fitted to the training data and scales both the training
    and testing data, and now the weights stay generally between +/- 10.0\n
    
    I wanted to see which features were most important for the classification, so I wrote a method for the Logistic
    Regression class which returns a list of tuples. Each tuple looks like (<feature-name>, <associated-weight>),
    which essentially shows the relative importance of the features. The resulting list is sorted from most to least
    important.\n
    
    Additionally, I wanted to tune some of the hyper-parameters. I randomly sampled 5000 logistic regression models
    with varying learning rates and learning rate diminishing functions and found that _____TODO_____\n
    
    
    Finally, I ran a preprocessing cascade on the raw text data to try to extract more information ____TODO_____
    """
        return s
    

def main():
    training = sys.argv[1]
    testing = sys.argv[2]
    training_tuples = generate_tuples_from_file(training)
    testing_tuples = generate_tuples_from_file(testing)
    
    # Pre-process text
    training_tuples = preprocess(training_tuples)
    testing_tuples = preprocess(testing_tuples)
    
    # Create our classifiers
    models = [SentimentAnalysis(), SentimentAnalysisImproved()]
    
    for model in models:
        model.train(training_tuples)
        p, r, f = evaluate_model(model, testing_tuples)
        print("Experiments:\n")
        print(model.describe_experiments())
        
        
def main2():
    training = sys.argv[1]
    testing = sys.argv[2]
    training_tuples = generate_tuples_from_file(training)
    testing_tuples = generate_tuples_from_file(testing)
    
    # Pre-process text
    training_tuples = preprocess(training_tuples)
    testing_tuples = preprocess(testing_tuples)
    
    
    
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
        sys.exit(1)
    
    main()
