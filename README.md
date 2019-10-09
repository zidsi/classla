# StanfordNLP: A Python NLP Library for Many Human Languages
## A [CLASSLA](http://www.clarin.si/info/k-centre/) Fork For Processing South Slavic Languages 

This is a fork of the [Stanford NLP Group's official Python NLP library](https://github.com/stanfordnlp/stanfordnlp). The main changes to the official library are
- possibility of training a tagger on training datasets which are not UD-parsed (regular situation for most languages)
- using an external dictionary while performing lemmatization
- looking up (word, upos, feats) triples in dictionaries during lemmatization, fallbacking to (word, upos)
- speeding up the seq2seq training procedure for lemmatization by training only on instances occurring less than 5 times in the training data
- speeding up the lemmatization (seq2seqing only forms not seen in the training data and lexicon) also outside the pipeline mode
- encoding POS information at the beginning of the sequence, improving significantly seq2seq lemmatization results

## Running the tool

### Part-of-speech tagging

Once you placed the Slovene PoS-tagging model (files ```ssj500``` and ```ssj500k.pretrain.pt```) into the ```models/pos/``` path, you can run the following command:

```
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name ssj500k --eval_file data/ssj500k.dev.conllu --output_file temp --gold_file data/ssj500k.dev.conllu --shorthand sl_ssj --mode predict
```

Similarly, you can try out the Croatian and Serbian tagger as well:

```
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name hr500k --eval_file data/hr500k.dev.conllu --output_file temp --gold_file data/hr500k.dev.conllu --shorthand sl_ssj --mode predict
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name SETimes.SR --eval_file data/SETimes.SR.dev.conllu --output_file temp --gold_file data/SETimes.SR.dev.conllu --shorthand sl_ssj --mode predict
```

### Lemmatization

```
python -m stanfordnlp.models.lemmatizer --model_dir models/lemma/ --model_file ssj500k+sloleks --eval_file out/pos.ssj500k.test.conllu --output_file out/lemma.ssj500k+lex.test.conllu --gold_file out/pos.ssj500k.test.conllu --mode predict
```

### Parsing

## Training your own models

### Part-of-speech tagging

Below are examples for training models for standard Slovene, Croatian and Serbian, assuming (1) CLARIN.SI embeddings and (2) corresponding training datasets (ssj500k, hr500k, SETimes.SR) are in the specified locations. Training is performed on train+test data, while dev is used for evaluation.

```
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name ssj500k --wordvec_file ~/data/clarin.si-embed/embed.sl-token.ft.sg.vec.xz --train_file ../babushka-bench/datasets/sl/ssj500k/train+test.conllu --eval_file ../babushka-bench/datasets/sl/ssj500k/dev.conllu --gold_file ../babushka-bench/datasets/sl/ssj500k/dev.conllu --mode train --shorthand sl_ssj --output_file temp.sl
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name hr500k --wordvec_file ~/data/clarin.si-embed/embed.hr-token.ft.sg.vec.xz --train_file ../babushka-bench/datasets/hr/hr500k/train+test.conllu --eval_file ../babushka-bench/datasets/hr/hr500k/dev.conllu --gold_file ../babushka-bench/datasets/hr/hr500k/dev.conllu --mode train --shorthand sl_ssj --output_file temp.hr
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name SETimes.SR --wordvec_file ~/data/clarin.si-embed/embed.sr-token.ft.sg.vec.xz --train_file ../babushka-bench/datasets/sr/SETimes.SR/train+test.conllu --eval_file ../babushka-bench/datasets/sr/SETimes.SR/dev.conllu --gold_file ../babushka-bench/datasets/sr/SETimes.SR/dev.conllu --mode train --shorthand sl_ssj --output_file temp.sr
```

### Lemmatization

Given that the lemmatization process relies on XPOS, to make the training data as realistic as possible, we train the lemmatization process on automatically part-of-speech-tagged data. We also use external lexicons for lemmatization (Sloleks, hrLex and srLex).

The pre-tagging is performed as defined in the section of running part-of-speech tagging. The commands for training the lemmatizers are given below.

```
python -m stanfordnlp.models.lemmatizer --model_dir models/lemma/ --model_file ssj500k+Sloleks --train_file pretagged/pos.ssj500k.train+test.conllu --eval_file pretagged/pos.ssj500k.dev.conllu --output_file temp.lem.sl --gold_file pretagged/pos.ssj500k.dev.conllu --external_dict ~/data/morphlex/Sloleks --mode train --num_epoch 30 --decay_epoch 20 --pos
python -m stanfordnlp.models.lemmatizer --model_dir models/lemma/ --model_file hr500k+hrLex --train_file pretagged/pos.hr500k.train+test.conllu --eval_file pretagged/pos.hr500k.dev.conllu --output_file temp.lem.hr --gold_file pretagged/pos.hr500k.dev.conllu --external_dict ~/data/morphlex/hrLex --mode train --num_epoch 30 --decay_epoch 20 --pos
python -m stanfordnlp.models.lemmatizer --model_dir models/lemma/ --model_file SETimes.SR+srLex --train_file pretagged/pos.SETimes.SR.train+test.conllu --eval_file pretagged/pos.SETimes.SR.dev.conllu --output_file temp.lem.sr --gold_file pretagged/pos.SETimes.SR.dev.conllu --external_dict ~/data/morphlex/srLex --mode train --num_epoch 30 --decay_epoch 20 --pos
```
