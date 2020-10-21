import numpy as np
import features
from types import FunctionType
import re
from math import log


"""
HELPER FUNCTIONS
"""

def label_map(neg_label='0', pos_label='1'):
    return {
        neg_label: 0,
        pos_label: 1
    }


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

def add_negation(data):
    """Adds negation information to the given data.
    Specifically, for every sentence that contains a negation word (ie. not, non),
    then this function will prepend NEGATIVE_ onto the beginning of every other word
    in the sentence.
    Parameters:
        data (str) - The raw text data
    Returns:
        str - The negated data
    """
    #
    # temp_data = data
    # current_idx = 0
    # final_data = ""
    # # regular_exp = r'([N|n]ot|[N|n]o|[N|n]ever)+([^.!?\n])*[.!?\n]'
    # regular_exp = r'(?<=[N|n]ot|[N|n]o|[N|n]ever) +([\w* *])*(?=|[A-Z]|$)'
    #
    #
    # current_match = re.search(regular_exp, temp_data)
    #
    # while current_match:
    #     final_data += temp_data[current_idx : current_match.start(0) - 1]
    #     current_match = re.search(regular_exp, )
    #
    # match_iterator = regular_exp.finditer(temp_data)
    # for match in match_iterator:
    #     print(match.group(0))
    #     # subset_of_data = temp_data[match.start(0): match.end(0)]
    #     # for word in range(len(subset_of_data.split())):
    #     #     if word != 0:
    #     #         subset_of_data[word] = "NEGATIVE_" + subset_of_data[word]
    #     # data[match.start(0): match.end(0)] = subset_of_data
        
    return data
    
    
def preprocess(all_examples):
    """Runs the following preprocessing steps on the data:
    1. all to lowercase
    2. TODO: add negation prefixes
    Parameters:
        all_examples (list of tuples) - All the training examples
    Returns:
        list of tuples - the cleaned data
    """
    all_lowercase = [(tup[0], tup[1].lower(), tup[2]) for tup in all_examples]
    
    all_negation = []
    for tup in all_lowercase:
        negated = add_negation(tup[1])
        all_negation.append((tup[0], negated, tup[2]))
    
    return all_negation


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


def cross_entropy_loss(y, sig):
    """Returns the cross-entropy loss from the given values
    Parameters:
        y (number) - The true label, either 0 or 1
        sig (number) - The estimated y value, the output of the sigmoid
    Returns:
        number - the cross-entropy loss
    """
    return -((y * log(sig)) + (1 - y) * log(1 - sig))


def feature_iterator():
    """Iterator for the callable objects of the features module
    Parameters:
        None
    Yields:
        function - the next feature function
    """
    for name, val in features.__dict__.items():
        if isinstance(val, FunctionType):
            yield name, val


def sigmoid(z):
    """Calculate the sigmoid of the input value z
    Parameters:
        z (number) - input value
    Returns:
        number - Sigmoid(z)
    """
    return pow(1 + np.exp(-z), -1)


class MinMaxScaler:
    """
    This class runs min-max scaling on training data, and can then
    scale any of the testing data with the trained vectors.
    """
    
    def __init__(self):
        self.array_length = None
        self.min_arr = None
        self.max_arr = None

    
    def fit(self, vects):
        """Build up the scaling vectors; We want to create a minimum and maximum
        vector, in which each entry is the minimum across all other vectors'
        corresponding entries
        Parameters:
            vects (list of arrays) - The list of feature vectors
        Returns:
            None
        """
        self.array_length = vects[0].shape[0]
        
        self.min_arr = [float('inf') for _ in range(self.array_length)]
        self.max_arr = [-float('inf') for _ in range(self.array_length)]
        
        for arr in vects:
            for idx in range(self.array_length):
                self.min_arr[idx] = min(self.min_arr[idx], arr[idx])
                self.max_arr[idx] = max(self.max_arr[idx], arr[idx])
        
        self.min_arr = np.array(self.min_arr)
        self.max_arr = np.array(self.max_arr)
        
        self.denom = np.subtract(self.max_arr, self.min_arr)
        # If any of the values are 0, replace with 1
        # This won't affect the math, we just won't get any DivByZero errors
        self.denom[self.denom == 0] = 1
        
        
    def scale(self, vect):
        """Normalizes the data in vect
        Parameters:
            vect (np.array) - the array to min-max scale
        Returns:
            np.array - the scaled vector
        """
    
        numer = np.subtract(vect, self.min_arr)
        return np.divide(numer, self.denom)
    

"""
MODEL EVALUATION FUNCTIONS
"""
def precision(gold_labels, classified_labels, pos_label='1', neg_label='0'):
    """Calculates the precision score given the true labels and the classified labels.
    Parameters:
        gold_labels (list) - The true labels of the data
        classified_labels (list) - The classified output of our model
        pos_label (str) - The label assigned as positive
        neg_label (str) - The label assigned as negative
    Returns:
        float representing the precision metric of our classification
    """
    # precision = tp/(tp + fp)
    true_positives = 0
    false_positives = 0
    
    for i in range(len(gold_labels)):
        if gold_labels[i] == pos_label and classified_labels[i] == pos_label:
            true_positives += 1
        elif gold_labels[i] == neg_label and classified_labels[i] == pos_label:
            false_positives += 1
            
    if true_positives + false_positives == 0:
        return 0
    
    return true_positives / (true_positives + false_positives)


def recall(gold_labels, classified_labels, pos_label='1', neg_label='0'):
    """Calculates the recall score given the true labels and the classified labels.
    Parameters:
        gold_labels (list) - The true labels of the data
        classified_labels (list) - The classified output of our model
        pos_label (str) - The label assigned as positive
        neg_label (str) - The label assigned as negative
    Returns:
        float representing the recall metric of our classification
    """
    # recall = tp/(tp + fn)
    
    true_positives = 0
    false_negatives = 0
    
    for i in range(len(gold_labels)):
        if gold_labels[i] == pos_label and classified_labels[i] == pos_label:
            true_positives += 1
        elif gold_labels[i] == pos_label and classified_labels[i] == neg_label:
            false_negatives += 1
            
    if true_positives + false_negatives == 0:
        return 0
    
    return true_positives / (true_positives + false_negatives)


def f1(gold_labels, classified_labels, pos_label='1', neg_label='0'):
    """Calculates the F1 score given the true labels and the classified labels.
    Parameters:
        gold_labels (list) - The true labels of the data
        classified_labels (list) - The classified output of our model
        pos_label (str) - The label assigned as positive
        neg_label (str) - The label assigned as negative
    Returns:
        float representing the F1 metric of our classification
    """
    # f1 = (2 * pr * re) / (pr + re)
    prec = precision(gold_labels, classified_labels, pos_label, neg_label)
    rec = recall(gold_labels, classified_labels, pos_label, neg_label)
    
    if prec + rec == 0:
        return 0
    
    return (2 * prec * rec) / (prec + rec)


def evaluate_model(model, dev_set):
    """Calculates all related scores for the model and prints them out, along with a description of the model
    Parameters:
        model (object) - The classifier to evaluate
        dev_set (list of tuples) - The development set to test the model on
    Returns:
        tuple - (precision, recall, f1-score) of the model on the given dev set
    """
    classified_labels = []
    gold_labels = []
    for test in dev_set:
        classified_labels.append(model.classify(test[1]))
        gold_labels.append(test[2])
    
    print(f"gold: {gold_labels}")
    print(f"class: {classified_labels}")
    
    precision_score = precision(gold_labels, classified_labels)
    recall_score = recall(gold_labels, classified_labels)
    f1_score = f1(gold_labels, classified_labels)
    
    print(f"Model output for: {model}")
    print(f"Precision: {precision_score}")
    print(f"Recall: {recall_score}")
    print(f"F1 metric: {f1_score}")
    
    return precision_score, recall_score, f1_score
