import os.path
import pandas as pd
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader

class NYT10Dataset(Dataset):
    """NYT10 dataset"""

    def __init__(self, sentence_file, relation2id_file):
        """
        Args:
            sentence_file (string): Path to the txt file with annotations.
            relation2id_file (string): Path to txt file with mapping between relation and id
        """
        self.clean_and_load_dataset(sentence_file)

        self.relation2id = {}
        with open(relation2id_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                self.relation2id[row[0]] = int(row[1])

    def clean_and_load_dataset(self, sentence_file):
        cleaned_sentence_file = sentence_file[:-4] + '_cleaned' + sentence_file[-4:]
        if os.path.isfile(cleaned_sentence_file):
            print('Cleaned file found! Loading now...')
            self.sentences_frame = pd.read_csv(cleaned_sentence_file, sep='\t', keep_default_na=False)
            print('Number of trainable samples:', len(self.sentences_frame))
            return

        print('No cleaned file found, cleaning now...')

        colnames = ['fb_mid_e1', 'fb_mid_e2', 'e1_name', 'e2_name', 'relation', 'sentence', 'end']
        self.sentences_frame = pd.read_csv(sentence_file,
                                           sep='\t',
                                           names=colnames,
                                           keep_default_na=False)
        self.sentences_frame.drop('end', axis=1, inplace=True)
        print('Number of samples loaded:', len(self.sentences_frame))

        broken_samples = []
        for i, (_, _, e1_name, e2_name, _, text) in self.sentences_frame.iterrows():
            split_text = text.split(' ')
            if e1_name not in split_text or e2_name not in split_text:
                broken_samples.append(i)
        print('Number of broken samples:', len(broken_samples))

        self.sentences_frame.drop(broken_samples, axis=0, inplace=True)
        print('Number of trainable samples:', len(self.sentences_frame))

        self.sentences_frame.to_csv(cleaned_sentence_file, sep='\t', index=False)

    def __len__(self):
        return len(self.sentences_frame)

    def __getitem__(self, idx):
        _, _, e1_name, e2_name, relation, text = self.sentences_frame.iloc[idx]
        text = text.split(' ')
        e1_index = text.index(e1_name)
        e2_index = text.index(e2_name)
        lower, upper = sorted((e1_index, e2_index))

        c1 = text[:lower]
        c2 = text[lower+1:upper]
        c3 = text[upper+1:]
        return c1, c2, c3, self.relation2id[relation]

if __name__ == 'main':
    sentences_dataset = NYT10Dataset('data/test.txt', 'data/relation2id.txt')
    print(sentences_dataset.sentences_frame.head())
    print(sentences_dataset.relation2id)