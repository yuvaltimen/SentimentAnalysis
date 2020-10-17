from nltk import sent_tokenize
"""
Yuval Timen
Module containing functions to extract features from data
Each function represents a single feature
"""

def num_sentences(data):
    """Finds the number of sentences in the data
    Parameters:
        data (str) - The raw text data, ie. the middle element of the training tuple
    Returns:
        int - the number of sentences in the data
    """
    return len(sent_tokenize(data))


def num_exclamation_marks(data):
    """Finds the number of exclamation points in the data
    Parameters:
        data (str) - The raw text data, ie. the middle element of the training tuple
    Returns:
        int - the number of "!" in the data
    """
    return data.count("!")


def get_count_words_in_list(word_list):
    """Returns the function that counts the number of words in the given list
    Parameters:
        word_list (list) - The list of words to reference
    Returns:
        function - The function that takes in data and counts the number of
        occurrences in the word_list
    """
    return lambda data: count_words_in_list(data, word_list)


def count_words_in_list(data, word_list):
    """Finds the number of words in the data that also appear in the list
    Parameters:
        data (str) - The raw text data, ie. the middle element of the training tuple
        word_list (list) - the list of words to reference
    Returns:
        int - the number of words in the data that appear in the list
    """
    count = 0
    all_words = data.split()
    for word in all_words:
        if word in word_list:
            count += 1
    return count


def bias_term(data):
    """Returns 1 to incorporate the bias term into the weight matrix.
    Parameters:
        data (str) - The raw text data, ie. the middle element of the training tuple
    Returns:
        int - the number 1; will be dotted with the bias term during training
    """
    return 1
