# StanfordNLP: A Python NLP Library for Many Human Languages
## A [CLASSLA](http://www.clarin.si/info/k-centre/) Fork For Processing South Slavic Languages 
[![Travis Status](https://travis-ci.com/stanfordnlp/stanfordnlp.svg?token=RPNzRzNDQRoq2x3J2juj&branch=master)](https://travis-ci.com/stanfordnlp/stanfordnlp)
[![PyPI Version](https://img.shields.io/pypi/v/stanfordnlp.svg?colorB=blue)](https://pypi.org/project/stanfordnlp/)
![Python Versions](https://img.shields.io/pypi/pyversions/stanfordnlp.svg?colorB=blue)

This is a fork of the [Stanford NLP Group's official Python NLP library](https://github.com/stanfordnlp/stanfordnlp). The main changes to the official library are
- possibility of training a tagger on training datasets which were not UD-parsed (in full)
- using an external dictionary while performing lemmatization
- speeding up the lemmatization (seq2seqing only forms not seen in the training data) also outside the pipeline mode

## Running the tool

### Part-of-speech tagging

Once you placed the Slovene PoS-tagging model (files ```ssj500``` and ```ssj500k.pretrain.pt```) into the ```models/pos/``` path, you can run the following command:

```
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name ssj500k --eval_file data/ssj500k.test.conllu --output_file temp --gold_file data/ssj500k.test.conllu --shorthand sl_ssj --mode predict
```

## Training your own models

### Part-of-speech tagging

```
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name ssj500k --wordvec_file ~/data/clarin.si-embed/embed.sl-token.ft.sg.vec.xz --train_file ../babushka-bench/datasets/sl/ssj500k/train.conllu --eval_file ../babushka-bench/datasets/sl/ssj500k/dev.conllu --gold_file ../babushka-bench/datasets/sl/ssj500k/dev.conllu --batch_size 5000 --mode train --shorthand sl_ssj --output_file temp
```
