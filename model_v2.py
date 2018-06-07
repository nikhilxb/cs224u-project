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
    def __init__(self, vocab, embed_dim, output_dim, position_dim=5, hidden_dim=230, dropout=0.5, num_positions=200):
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

        self.position_dim = position_dim
        self.num_positions = num_positions
        self.positions = nn.Embedding(2*num_positions+1, position_dim)

        self.pcnn = PiecewiseCNN(embed_dim + 2 * position_dim, hidden_dim)
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
        mid_lens = [len(c2) for c1, c2, c3 in X]
        batch_C1, batch_C2, batch_C3 = zip(*X)  # list[ tuple(1_c1, 1_c2, 1_c3), tuple(2_c1, 2_c2, 2_c3), ...]

        # converts a batch of sentences (word lists) into a single, length-padded tensor
        def to_tensor(batch_c, seq_pos):
            vec_seqs = [self._assemble_vec_seq(c, mid_lens[i], seq_pos) for i, c in enumerate(batch_c)]
            padded_seqs = RelationClassifier._pad_sequence_unsorted(vec_seqs, batch_first=True)
            padded_seqs = padded_seqs.transpose(1, 2)
            return padded_seqs

        # output each is FloatTensor(batch_size, embed_dim, max_piece_len)
        padded_C1, padded_C2, padded_C3 = to_tensor(batch_C1, 1), to_tensor(batch_C2, 2), to_tensor(batch_C3, 3)

        h = self.pcnn(padded_C1, padded_C2, padded_C3)  # (batch_size, output_dim, 3)
        batch_size = h.size()[0]
        h = h.view(batch_size, -1)  # (batch_size, 3 * output_dim)
        h = self.drop1(h)
        h = self.lin1(h)
        return h

    def _assemble_vec_seq(self, seq, mid_len, seq_pos):
        """
        Args:
            seq = list[ string, ... ] = list of word strings
            mid_len = length of middle segment
        Return:
            FloatTensor(seq_len, embed_dim + 2 * position_dim)
        """
        # Assemble word vecs
        word_vecs = [self.vocab.get(word, self.unk) for word in seq] if len(seq) > 0 else [torch.zeros(self.unk.shape)]
        word_vecs = torch.stack(word_vecs)

        # Assemble pos vecs
        pos1 = torch.arange(0, len(seq), dtype=torch.long)
        pos2 = torch.arange(0, len(seq), dtype=torch.long)
        if seq_pos == 1:
            pos1 = pos1 - len(seq)
            pos2 = pos2 - mid_len - len(seq) - 1
        elif seq_pos == 2:
            pos1 = pos1 + 1
            pos2 = pos2 - mid_len
        elif seq_pos == 3:
            pos1 = pos1 + mid_len + 2
            pos2 = pos2 + 1
        pos1 = torch.clamp(pos1, -self.num_positions, self.num_positions) + self.num_positions  # scale so (0, 2*num_positions+1)
        pos2 = torch.clamp(pos2, -self.num_positions, self.num_positions) + self.num_positions
        pos1_vecs = self.positions(pos1)  # (seq_len, position_dim)
        pos2_vecs = self.positions(pos2)

        return torch.cat((word_vecs, pos1_vecs, pos2_vecs), dim=1)

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
    rc = RelationClassifier(vocab, 50, 3, position_dim=2)
    X = [ (["first", "piece"], ["second", "piece"], ["third", "piece"]) ]
    y = [ 0 ]
    print("RelationClassifier test:")
    print("--- _assemble_vec_seq:\n", rc._assemble_vec_seq(['apple', 'banana', 'coconut', 'durian', 'apple'], 5, 1)[:, -5:])
    out = rc(X)
    print("--- forward:\n", out.size() == (1, 3))
