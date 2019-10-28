"""
Exceptions for machine learning
"""

class MachineLearningError(Exception):
    """Base exception class for machine learning"""
    pass

def raise_no_module_error(module):
    raise MachineLearningError("No \'{}\' installed in your machine.".format(module))