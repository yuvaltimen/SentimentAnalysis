import numpy as np
import sys
from math import log
import features

"""
Yuval Timen
Sentiment Analysis using Naive Bayes Classifier and Logistic Regression
0 represents a negative review, and 1 represents positive
"""

"""
Cite your sources here:
"""

"""
Implement your functions that are not methods of the Sentiment Analysis class here
"""
def k_fold(all_examples, k):
    """Splits the data into k folds, where each fold can be used to measure
    the performance of the model
    Parameters:
        all_examples (list of tuples) - The list of all tuples in our training data
        k (int) - the number of folds to split our data into
    Returns:
        list of lists - A list of lists, each of which contains the test-train split of the data
    """
    pass


def load_tokens_from_file(file_path_name):
    """Tokenizes the data in the given file path and returns the content as a list
    Parameters:
        file_path_name (str) - The location of the data
    Returns:
        list - A list of tokens
    """
    output = []
    with open(file_path_name, 'r') as read_file:
        for line in read_file:
            if line.strip() == "":
                break
            output.append(line.strip())
            
    return output
    

def generate_tuples_from_file(training_file_path):
    """Returns a list of tuples from the given training file path.
    All training data is formatted as follows: [ID]\t[Review]\t[Label]\n
    We can use this to separate our reviews into a list of tuples, each
    formatted like (id, example_text, label)
    Parameters:
        training_file_path (str) - The name of the training data file
    Returns:
        list of tuples representing the features of the data
    """
    output = []
    with open(training_file_path, 'r') as file_read:
        for line in file_read.readlines():
            line = line.strip()
            if len(line) != 0:
                output.append(tuple(line.split("\t")))
    
    print(f"Read-in {training_file_path[:10]}, found {len(output)} lines")
    return output


def precision(gold_labels, classified_labels):
    """Calculates the precision score given the true labels and the classified labels.
    Parameters:
        gold_labels (list) - The true labels of the data
        classified_labels (list) - The classified output of our model
    Returns:
        float representing the precision metric of our classification
    """
    # precision = tp/(tp + fp)
    true_positives = 0
    false_positives = 0
    
    for i in range(len(gold_labels)):
        if gold_labels[i] == '1' and classified_labels[i] == '1':
            true_positives += 1
        elif gold_labels[i] == '0' and classified_labels[i] == '1':
            false_positives += 1
    
    return true_positives / (true_positives + false_positives)


def recall(gold_labels, classified_labels):
    """Calculates the recall score given the true labels and the classified labels.
    Parameters:
        gold_labels (list) - The true labels of the data
        classified_labels (list) - The classified output of our model
    Returns:
        float representing the recall metric of our classification
    """
    # recall = tp/(tp + fn)
    
    true_positives = 0
    false_negatives = 0
    
    for i in range(len(gold_labels)):
        if gold_labels[i] == '1' and classified_labels[i] == '1':
            true_positives += 1
        elif gold_labels[i] == '1' and classified_labels[i] == '0':
            false_negatives += 1
    
    return true_positives / (true_positives + false_negatives)
    

def f1(gold_labels, classified_labels):
    """Calculates the F1 score given the true labels and the classified labels.
    Parameters:
        gold_labels (list) - The true labels of the data
        classified_labels (list) - The classified output of our model
    Returns:
        float representing the F1 metric of our classification
    """
    # f1 = (2 * pr * re) / (pr + re)
    prec = precision(gold_labels, classified_labels)
    rec = recall(gold_labels, classified_labels)
    
    return (2 * prec * rec) / (prec + rec)
    

class SentimentAnalysis:
    
    def __init__(self, use_binary=False):
        """Initialize the SentimentAnalysis class"""
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
        self.priors = {k: v/len(examples) for k,v in self.priors.items()}
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
        
    
    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"



class SentimentAnalysisImproved:

    def __init__(self, learning_rate=0.1, use_stemming=False, use_negation=False):
        """Initialize the Logistic Regression class"""
        self.learning_rate = learning_rate
        self.stemming = use_stemming
        self.negation = use_negation
        self.positive_words = load_tokens_from_file("positive_words.txt")
        self.negative_words = load_tokens_from_file("negative_words.txt")
        self.features = [
            features.num_sentences,
            features.num_exclamation_marks,
            features.get_count_words_in_list(self.positive_words),
            features.get_count_words_in_list(self.negative_words),
            features.bias_term
        ]
        self.weights = np.random.rand(len(self.features))


    def train(self, examples):
        """Train our Logistic Regression model by updating the weights in each iteration
        Uses stochastic gradient descent and the cross-entropy loss function to train
        Parameters:
            examples (list of tuples) - The list of tuples to train on
        Returns:
            None
        """
        X = np.array([self.featurize(example[1]) for example in examples])
        print(X)
        y = [example[2] for example in examples]
        
        
        
        
        


    def score(self, data):
        """Uses the model's weights to calculate a score for the given data point
        Parameters:
            data (str) - The raw text data, ie. the middle element in the tuple
        Returns:
            dict - a mapping from each class to the associated score
        """
        pass


    def classify(self, data):
        """Returns the predicted class of the given data point
        Parameters:
            data (str) - The raw text data, ie. the middle element in the tuple
        Returns:
            str - label, either '0' or '1'
        """
        pass


    def featurize(self, data):
        """Featurizes the data by returning a vector representation of the data
        Parameters:
            data (str) - The raw text data, ie. the middle element in the tuple
        Returns:
            np.array - the vector representation of the featurized data
        """
        ans = []
        for feature in self.features:
            ans.append(feature(data))
            
        return np.array(ans)
        

    def __str__(self):
        return "Logistic Regression classifier"


    def describe_experiments(self):
        s = """
    Description of your experiments and their outcomes here.
    """
        return s
    

def main():
    training = sys.argv[1]
    testing = sys.argv[2]
    training_tuples = generate_tuples_from_file(training)
    testing_tuples = generate_tuples_from_file(testing)
    
    # Create our classifier
    sa = SentimentAnalysis()
    print(sa)
    # Train the classifier on our training data
    sa.train(training_tuples)
    # Report metrics for the baseline
    classified_labels = []
    gold_labels = []
    for test in testing_tuples:
        classified_labels.append(sa.classify(test[1]))
        gold_labels.append(test[2])
    
    print(f"Precision: {precision(gold_labels, classified_labels)}")
    print(f"Recall: {recall(gold_labels, classified_labels)}")
    print(f"F1 metric: {f1(gold_labels, classified_labels)}")
    
    improved = SentimentAnalysisImproved()
    print(improved)
    # Train the classifier on our training data

    # report final precision, recall, f1 (for your best model)

    print(improved.describe_experiments())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
        sys.exit(1)
    
    main()
