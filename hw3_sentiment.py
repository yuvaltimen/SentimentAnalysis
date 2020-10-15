import numpy as np
import sys
from math import log

"""
Yuval Timen
Sentiment Analysis using Naive Bayes Classifier
0 represents a negative review, and 1 represents positive
"""

"""
Cite your sources here:
"""

"""
Implement your functions that are not methods of the Sentiment Analysis class here
"""


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


"""
Implement any other non-required functions here
"""

"""
implement your SentimentAnalysis class here
"""


class SentimentAnalysis:
    
    def __init__(self):
        """Initialize the SentimentAnalysis class"""
        self.priors = {'0': 0, '1': 0}
        self.positives_word_counts = {}
        self.negatives_word_counts = {}
        self.positives_denominator = 0
        self.negatives_denominator = 0
        self.vocabulary = set()
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
        print("Training...")
        print("Building vocabulary and updating priors...")
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
        
        print("Compiling documents...")
        # Next, we need to concatenate all documents with common class c
        positive_examples = list(filter(lambda tup: bool(int(tup[2])), examples))
        positives_document = [tup[1] for tup in positive_examples]
        positives_document = " ".join(positives_document)
        
        negative_examples = list(filter(lambda tup: not bool(int(tup[2])), examples))
        negatives_document = [tup[1] for tup in negative_examples]
        negatives_document = " ".join(negatives_document)
        
        print("Training on positives...")
        # Update the word counts for all words in the positive document
        for w in positives_document.split():
            if w in self.positives_word_counts.keys():
                self.positives_word_counts[w] += 1
            else:
                self.positives_word_counts[w] = 1
        
        self.positives_denominator = sum([val for val in self.positives_word_counts.values()])
        print(f"Pos denom: {self.positives_denominator}")
        
        print("Training on negatives...")
        # Update the word counts for all words in the negative document
        for w in negatives_document.split():
            if w in self.negatives_word_counts.keys():
                self.negatives_word_counts[w] += 1
            else:
                self.negatives_word_counts[w] = 1

        self.negatives_denominator = sum([val for val in self.negatives_word_counts.values()])
        print(f"Neg denom: {self.negatives_denominator}")
        
        print("Training done!")
    
    
    def score(self, data):
        """Uses the models weights to calculate a score for the given data point
        Parameters:
            data (str) - The raw text data, ie. the middle element of the training tuple
        Returns:
            dict of values assigning to each class a probability
        """
        print("Scoring...")
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
        print("Featurizing...")
        output = []
        for w in data.split():
            output.append((w, True))
        return output
        
    
    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"



class SentimentAnalysisImproved:

    def __init__(self):
        pass

    def train(self, examples):
        pass

    def score(self, data):
        pass

    def classify(self, data):
        pass

    def featurize(self, data):
        pass

    def __str__(self):
        return "NAME OF YOUR CLASSIFIER HERE"

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
    
    
    
    
    #
    # improved = SentimentAnalysisImproved()
    # print(improved)
    # # do the things that you need to with your improved class
    #
    # # report final precision, recall, f1 (for your best model)
    #
    # print(improved.describe_experiments())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
        sys.exit(1)
    
    main()
