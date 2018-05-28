# CS 224U Final Project: Relation Extraction
Relation extraction through distal supervision with CNNs. Reproducing Zeng et al's work here: http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP203.pdf

## Data
Download from: https://github.com/thunlp/NRE/blob/master/data.zip and extract to the top-most level of the repository. Structure should be:

```
└── data/
      ├── entity2id.txt
      ├── relation2id.txt
      ├── test.txt
      ├── train.txt
      └── vec.bin
```

The dataset is in the following format:
+ train.txt: training file, format (fb_mid_e1, fb_mid_e2, e1_name, e2_name, relation, sentence).
+ test.txt: test file, same format as train.txt.
+ entity2id.txt: all entities and corresponding ids, one per line.
+ relation2id.txt: all relations and corresponding ids, one per line.
+ vec.bin: the pre-train word embedding file
