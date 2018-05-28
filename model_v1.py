import torch
import torch.autograd as autograd
import torch.nn as nn

class PiecewiseCNN(nn.Module):
    """Module to convert a sentence of word-vectors into a classification over relation labelsself.

    A text sentence S is decomposed in the following form:
        S = [C1] + [A] + [C2] + [B] + [C3]
    where A and B are the two entity mentions in a sentence and C1, C2, C3 are the before, midddle, after sequences of
    context words.

    In: list[ tuple(X, y), ... ]        list of training examples
       X = tuple(c1, a, c2, b, c3)      single training example
       y = int                          relation label index
       c1, c2, c3 = list[ w1, ... ]     list of words (strings)
    """
