# CS 224U Final Project: Relation Extraction
Relation extraction through distal supervision with CNNs. Reproducing Zeng et al's work here: http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf

## Data
Download the NYT10 relation extraction dataset from: https://github.com/thunlp/NRE/blob/master/data.zip and extract to the top-most level of the repository. Also download the Wikipedia 2014 + Gigaword 5 distribution of the pretrained GloVe vectors from here: http://nlp.stanford.edu/data/glove.6B.zip. Structure should be:

```
└── data/
      ├── entity2id.txt
      ├── relation2id.txt
      ├── test.txt
      ├── train.txt
      └── glove.6B.50d.txt
```

The dataset is in the following format:
+ train.txt: training file, format (fb_mid_e1, fb_mid_e2, e1_name, e2_name, relation, sentence).
+ test.txt: test file, same format as train.txt.
+ entity2id.txt: all entities and corresponding ids, one per line.
+ relation2id.txt: all relations and corresponding ids, one per line.
+ glove.6B.50d.txt: the pre-trained GloVe word embedding file hosted by Stanford NLP
