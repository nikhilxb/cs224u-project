import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import utils

class RelationClassifier(nn.Module):
    """Module to convert a sentence of word-vectors into a vector of predicted relation labels.

    A text sentence S is decomposed in the following form:
        S = [C1] + [A] + [C2] + [B] + [C3]
    where A and B are the two entity mentions in a sentence and C1, C2, C3 are the before, middle, after sequences of
    context words.
    """
    def __init__(self, vocab, embed_dim, output_dim, hidden_dim=230, dropout=0.5):
        """
        Args:
            vocab = dict[word] -> numpy array(embed_dim,) = vocabulary dict
            embed_dim = int                               = vocabulary embeddings dim
        """
        super(RelationClassifier, self).__init__()
        self.vocab = vocab
        for word, vec in self.vocab.items():
            self.vocab[word] = torch.FloatTensor(vec)
        self.embed_dim = embed_dim
        self.unk = torch.rand(self.embed_dim)

        self.pcnn = PiecewiseCNN(self.embed_dim, hidden_dim)
        self.drop1 = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(3 * hidden_dim, output_dim)

    def forward(self, X):
        """
        Args:
            X = list[ tuple(c1, c2, c3), ... ] = batch of training examples
            c1/c2/c3 = list[ string, ... ]     = list of word strings
        Return:
            FloatTensor(batch_size, num_labels) = predicted labels for each example in batch
        """
        batch_C1, batch_C2, batch_C3 = zip(*X)  # list[ tuple(1_c1, 1_c2, 1_c3), tuple(2_c1, 2_c2, 2_c3), ...]

        padded_C1 = RelationClassifier._pad_sequence_unsorted([self._assemble_vec_seq(c1) for c1 in batch_C1], batch_first=True).transpose(1, 2)
        padded_C2 = RelationClassifier._pad_sequence_unsorted([self._assemble_vec_seq(c2) for c2 in batch_C2], batch_first=True).transpose(1, 2)
        padded_C3 = RelationClassifier._pad_sequence_unsorted([self._assemble_vec_seq(c3) for c3 in batch_C3], batch_first=True).transpose(1, 2)
        # output each is FloatTensor(batch_size, embed_dim, max_piece_len)

        h = self.pcnn(padded_C1, padded_C2, padded_C3)  # (batch_size, output_dim, 3)
        batch_size = h.size()[0]
        h = h.view(batch_size, -1)  # (batch_size, 3 * output_dim)
        h = self.drop1(h)
        h = self.lin1(h)
        return h

    def _assemble_vec_seq(self, seq):
        """
        Args:
            seq = list[ string, ... ] = list of word strings
        Return:
            FloatTensor(seq_len, embed_dim)
        """
        x = [self.vocab.get(word, self.unk) for word in seq]
        if len(x) == 0:
            x = [torch.zeros(self.unk.shape)]
        return torch.stack(x)

    @staticmethod
    def _pad_sequence_unsorted(batch_c, batch_first=False):
        # create a numpy array indicating the original positions of C1
        positions = np.arange(len(batch_c))
        # sort the array based on the lengths of sequences of C1
        positions = [x for _, x in sorted(zip([len(c) for c in batch_c],positions), reverse=True)]
        # actually sort C
        batch_c.sort(key=lambda x:len(x), reverse=True)
        # pad it(Commented out here because I dont have torch)
        batch_c = pad_sequence(batch_c, batch_first=batch_first)
        # Sort it back to its original sequence(because we want it to match with C2, C3 etc)
        batch_c = [x for _,x in sorted(zip(positions, batch_c), reverse=False)]
        batch_c = torch.stack(batch_c)
        return batch_c


class PiecewiseCNN(nn.Module):
    """Applies convolution over a variable number of pieces and concatenates all the output pieces."""
    def __init__(self, input_dim, output_dim, kernel_size=3, padding=2):
        super(PiecewiseCNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, padding=padding)

    def forward(self, *pieces):
        """
        Args:
            pieces = list[ FloatTensor(batch_size, input_dim, piece_len) ] = tensors for each context piece
        Return:
            FloatTensor(batch_size, output_dim, num_pieces)
        """
        def process(x):  # (batch_size, input_dim, piece_len)
            x = self.conv1(x)  # (batch_size, output_dim, piece_len)
            x, _ = x.max(dim=2)  # (batch_size, output_dim)
            x = torch.tanh(x)
            return x

        processed = [process(piece) for piece in pieces]
        return torch.stack(processed, dim=2)  # (batch_size, output_dim, num_pieces)


if __name__ == "__main__":
    # Test PiecewiseCNN
    c1 = torch.rand(3, 4, 7, requires_grad=True)  # (batch_size, embed_size, sequence_len)
    c2 = torch.rand(3, 4, 2, requires_grad=True)
    pcnn = PiecewiseCNN(4, output_dim=10)
    out = pcnn(c1, c2)  # (batch_size, output_dim, num_pieces)
    print("PiecewiseCNN test:")
    print("--- out.size() == (3, 10, 2):", out.size() == (3, 10, 2))
    print()

    # Test RelationClassifier
    vocab = utils.glove2dict("data/glove.6B.50d.txt")  # dict[word] -> numpy array(embed_dim,)
    rc = RelationClassifier(vocab, 50)
    X = [ (["first", "piece"], ["second", "piece"], ["third", "piece"]) ]
    y = [ 0 ]
    print("RelationClassifier test:")
    print("--- _assemble_vec_seq:\n", rc._assemble_vec_seq(['apple', 'banana', 'coconut', 'durian', 'apple'])[:, :3])
    out = rc(X, y)
    print("--- forward:\n", out.size())
