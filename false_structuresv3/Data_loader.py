from abc import ABC, abstractmethod


class Data_loader:
    """Abstract class used for data loading.

Methods to be implemented by subclasses
---------------------------------------
load_data(): Loads data and labels 

Attibuts
--------
arguments (dict): Dictionary with all the arguments spesific for the subclass
"""

    def __init__(self, arguments):
        """ Superclass constructor
Arguments
---------
arguments (dict): Dictionary with all the arguments spesific for the subclass
"""
        self.arguments = arguments

    def _check_for_valid_arguments(self, keys_required, arguments):
        """Interates through the dictonary `arguments` and check 
that each of the keys in the list `keys_required` are keys in `arguments`

Arguments
---------
keys_required (list): List of strings representing requied keys in arguments
arguments (dict): Dictonary
        
Returns
-------
Nothing, but it raises an KeyError exception if not all required keys are present.
"""
        for req_key in keys_required:
            if req_key not in arguments:
                raise KeyError('Data loader did not find key: %s among it arguments' % req_key)

    @abstractmethod
    def load_data(self, x):
        """Loads the data as two numpy arrays x and y, where x is the 
        input data, and y are the labels
        """
        pass

