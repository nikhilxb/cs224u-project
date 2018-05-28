import torch
import torch.autograd as autograd
import torch.nn as nn

class RelationClassifier(nn.Module):
    """Module to convert a sentence of word-vectors into a classification over relation labels.

    A text sentence S is decomposed in the following form:
        S = [C1] + [A] + [C2] + [B] + [C3]
    where A and B are the two entity mentions in a sentence and C1, C2, C3 are the before, midddle, after sequences of
    context words.
    """
    def __init__(self,
                 word_vecs # (vocab_size, embed_dim)
                 ):
        """
        """
        self.word_vecs = word_vecs
        self.vocab_size, self.embed_dim = word_vecs.size()

    def forward(self, X, y):
        """
        In:
            X = list[ tuple(c1, a, c2, b, c3), ... ] = bag of training examples
            y = int                                  = relation label index of bag
            c1, c2, c3 = list[ w1, ... ]             = list of word indices
        """



class PiecewiseCNN(nn.Module):
    """Applies convolution over a variable number of pieces and concatenates all the output pieces."""
    def __init__(self,
                 input_dim,
                 output_dim=256,
                 kernel_size=3):
        self.conv1 = nn.Conv1d(input_dim, output_dim)

    def forward(self, *pieces):
        """
        In:
            pieces = list[ FloatTensor(batch_size, input_dim, piece_len) ] = tensors for each context piece
        Out:
            FloatTensor(batch_size, output_dim, num_pieces)
        """
        def process(in):
            out = self.conv1(in)
            out = out.max()
            return out

        processed = [self.conv1()]
