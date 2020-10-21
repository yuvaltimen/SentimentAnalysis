"""
Yuval Timen
Module containing functions to extract features from data
Each function represents a single feature
"""

class PreloadLists:
    """This class is for loading in the lists so that they don't have
    to be loaded in every time a word gets featurized. A function would not work
    because the feature_iterator in hw3_sentiment.py returns all callable
    objects from the module, and we want these lists to not be callable
    """
    positive_words = [word.strip() for word in open("positive_words.txt").readlines()]
    negative_words = [word.strip() for word in open("negative_words.txt").readlines()]


def count_positive_words_in_list(data):
    """Finds the number of words in the data that also appear in the list
    Parameters:
        data (str) - The raw text data, ie. the middle element of the training tuple
    Returns:
        int - the number of words in the data that appear in the list
    """
    from nltk import word_tokenize
    count = 0
    all_words = word_tokenize(data)
    for word in all_words:
        if word in PreloadLists.positive_words:
            count += 1
    return count


def count_negative_words_in_list(data):
    """Finds the number of words in the data that also appear in the list
    Parameters:
        data (str) - The raw text data, ie. the middle element of the training tuple
    Returns:
        int - the number of words in the data that appear in the list
    """
    from nltk import word_tokenize
    count = 0
    all_words = word_tokenize(data)
    for word in all_words:
        if word in PreloadLists.negative_words:
            count += 1
    return count


def count_words_in_data(data):
    """Finds the number of words in the data that also appear in the list
    Parameters:
        data (str) - The raw text data, ie. the middle element of the training tuple
    Returns:
        int - the number of words in the data
    """
    from nltk import word_tokenize
    return len(word_tokenize(data))


def count_numbers(data):
    """Determines if there is a numeric in the text data
    Parameters:
        data (str) - The raw text data, ie. the middle element of the training tuple
    Returns:
        bool - True if text has a number, false otherwise
    """
    import re
    return len(re.findall(r'[0-9]', data))
    
    
def count_exclamation_mark(data):
    """Finds the number of exclamation points in the data
    Parameters:
        data (str) - The raw text data, ie. the middle element of the training tuple
    Returns:
        int - the number of "!" in the data
    """
    import re
    return len(re.findall(r'!', data))


def count_uppercase_word(data):
    """Binary feature of whether there is a word that is upper-cased
    Parameters:
        data (str) - The raw text data, ie. the middle element of the training tuple
    Returns:
        int - 1 if true, 0 if false
    """
    count = 0
    for word in data.split():
        if word == word.upper():
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
