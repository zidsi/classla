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
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name hr500k --eval_file data/hr500k.dev.conllu --output_file temp --gold_file data/hr500k.dev.conllu --shorthand hr_set --mode predict
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name SETimes.SR --eval_file data/SETimes.SR.dev.conllu --output_file temp --gold_file data/SETimes.SR.dev.conllu --shorthand sr_set --mode predict
```

### Lemmatization

```
python -m stanfordnlp.models.lemmatizer --model_dir models/lemma/ --model_file ssj500k+Sloleks --eval_file pretagged/pos.ssj500k.dev.conllu --output_file pretagged/pos.lemma.ssj500k.dev.conllu --gold_file pretagged/pos.ssj500k.dev.conllu --mode predict

```

### Parsing

```
python -m stanfordnlp.models.parser --save_dir models/depparse/ --save_name ssj500k_ud --eval_file pretagged/pos.lemma.ssj500k_ud.dev.conllu --gold_file pretagged/pos.lemma.ssj500k_ud.dev.conllu --shorthand sl_ssj --output_file pretagged/pos.lemma.depparse.ssj500k_ud.dev.conllu --mode predict
```

## Training your own models

### Part-of-speech tagging

Below are examples for training models for standard Slovene, Croatian and Serbian, assuming (1) CLARIN.SI embeddings and (2) corresponding training datasets (ssj500k, hr500k, SETimes.SR) are in the specified locations. Training is performed on train+test data, while dev is used for evaluation.

```
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name ssj500k --wordvec_file ~/data/clarin.si-embed/embed.sl-token.ft.sg.vec.xz --train_file ../babushka-bench/datasets/sl/ssj500k/train+test.conllu --eval_file ../babushka-bench/datasets/sl/ssj500k/dev.conllu --gold_file ../babushka-bench/datasets/sl/ssj500k/dev.conllu --mode train --shorthand sl_ssj --output_file temp.sl
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name hr500k --wordvec_file ~/data/clarin.si-embed/embed.hr-token.ft.sg.vec.xz --train_file ../babushka-bench/datasets/hr/hr500k/train+test.conllu --eval_file ../babushka-bench/datasets/hr/hr500k/dev.conllu --gold_file ../babushka-bench/datasets/hr/hr500k/dev.conllu --mode train --shorthand hr_set --output_file temp.hr
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name SETimes.SR --wordvec_file ~/data/clarin.si-embed/embed.sr-token.ft.sg.vec.xz --train_file ../babushka-bench/datasets/sr/SETimes.SR/train+test.conllu --eval_file ../babushka-bench/datasets/sr/SETimes.SR/dev.conllu --gold_file ../babushka-bench/datasets/sr/SETimes.SR/dev.conllu --mode train --shorthand sr_set --output_file temp.sr
```

### Lemmatization

Given that the lemmatization process relies on XPOS, to make the training data as realistic as possible, we train the lemmatization process on automatically part-of-speech-tagged data. We also use external lexicons for lemmatization (Sloleks, hrLex and srLex).

The pre-tagging is performed as defined in the section of running part-of-speech tagging. The commands for training the lemmatizers are given below.

```
python -m stanfordnlp.models.lemmatizer --model_dir models/lemma/ --model_file ssj500k+Sloleks --train_file pretagged/pos.ssj500k.train+test.conllu --eval_file pretagged/pos.ssj500k.dev.conllu --output_file temp.lemma.sl --gold_file pretagged/pos.ssj500k.dev.conllu --external_dict ~/data/morphlex/Sloleks --mode train --num_epoch 30 --decay_epoch 20 --pos
python -m stanfordnlp.models.lemmatizer --model_dir models/lemma/ --model_file hr500k+hrLex --train_file pretagged/pos.hr500k.train+test.conllu --eval_file pretagged/pos.hr500k.dev.conllu --output_file temp.lemma.hr --gold_file pretagged/pos.hr500k.dev.conllu --external_dict ~/data/morphlex/hrLex --mode train --num_epoch 30 --decay_epoch 20 --pos
python -m stanfordnlp.models.lemmatizer --model_dir models/lemma/ --model_file SETimes.SR+srLex --train_file pretagged/pos.SETimes.SR.train+test.conllu --eval_file pretagged/pos.SETimes.SR.dev.conllu --output_file temp.lemma.sr --gold_file pretagged/pos.SETimes.SR.dev.conllu --external_dict ~/data/morphlex/srLex --mode train --num_epoch 30 --decay_epoch 20 --pos
```

### Parsing

Preparing the data to be automatically pretagged on morphosyntactic and lemma level:

```
# Slovene tagging
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name ssj500k --eval_file ../babushka-bench/datasets/sl/ssj500k/dev_ud.conllu --output_file pretagged/pos.ssj500k_ud.dev.conllu --gold_file ../babushka-bench/datasets/sl/ssj500k/dev_ud.conllu --shorthand sl_ssj --mode predict
cat ../babushka-bench/datasets/sl/ssj500k/train_ud.conllu ../babushka-bench/datasets/sl/ssj500k/test_ud.conllu > pretagged/ssj500k_ud.train+test.conllu
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name ssj500k --eval_file pretagged/ssj500k_ud.train+test.conllu --output_file pretagged/pos.ssj500k_ud.train+test.conllu --gold_file pretagged/ssj500k_ud.train+test.conllu --shorthand sl_ssj --mode predict
# Croatian tagging
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name hr500k --eval_file ../babushka-bench/datasets/hr/hr500k/dev_ud.conllu --output_file pretagged/pos.hr500k_ud.dev.conllu --gold_file ../babushka-bench/datasets/hr/hr500k/dev_ud.conllu --shorthand hr_set --mode predict
cat ../babushka-bench/datasets/hr/hr500k/train_ud.conllu ../babushka-bench/datasets/hr/hr500k/test_ud.conllu > pretagged/hr500k_ud.train+test.conllu
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name hr500k --eval_file pretagged/hr500k_ud.train+test.conllu --output_file pretagged/pos.hr500k_ud.train+test.conllu --gold_file pretagged/hr500k_ud.train+test.conllu --shorthand hr_set --mode predict
# Serbian tagging
python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name SETimes.SR --eval_file ../babushka-bench/datasets/sr/SETimes.SR/dev_ud.conllu --output_file pretagged/pos.SETimes.SR_ud.dev.conllu --gold_file ../babushka-bench/datasets/sr/SETimes.SR/dev_ud.conllu --shorthand sr_set --mode predict
cat ../babushka-bench/datasets/sr/SETimes.SR/train_ud.conllu ../babushka-bench/datasets/sr/SETimes.SR/test_ud.conllu > pretagged/SETimes.SR_ud.train+test.conllu
(pytorch) nikolal@kt-gpu-vm-1TB:~/tools/classla-stanfordnlp$ python -m stanfordnlp.models.tagger --save_dir models/pos/ --save_name SETimes.SR --eval_file pretagged/SETimes.SR_ud.train+test.conllu --output_file pretagged/pos.SETimes.SR_ud.train+test.conllu --gold_file pretagged/SETimes.SR_ud.train+test.conllu --shorthand sr_set --mode predict
# Slovene lemmatization
python -m stanfordnlp.models.lemmatizer --model_dir models/lemma/ --model_file ssj500k+Sloleks --eval_file pretagged/pos.ssj500k_ud.dev.conllu --output_file pretagged/pos.lemma.ssj500k_ud.dev.conllu --gold_file pretagged/pos.ssj500k_ud.dev.conllu --mode predict
python -m stanfordnlp.models.lemmatizer --model_dir models/lemma/ --model_file ssj500k+Sloleks --eval_file pretagged/pos.ssj500k_ud.train+test.conllu --output_file pretagged/pos.lemma.ssj500k_ud.train+test.conllu --gold_file pretagged/pos.ssj500k_ud.train+test.conllu --mode predict
# Croatian lemmatization
python -m stanfordnlp.models.lemmatizer --model_dir models/lemma/ --model_file hr500k+hrLex --eval_file pretagged/pos.hr500k_ud.dev.conllu --output_file pretagged/pos.lemma.hr500k_ud.dev.conllu --gold_file pretagged/pos.hr500k_ud.dev.conllu --mode predict
python -m stanfordnlp.models.lemmatizer --model_dir models/lemma/ --model_file hr500k+hrLex --eval_file pretagged/pos.hr500k_ud.train+test.conllu --output_file pretagged/pos.lemma.hr500k_ud.train+test.conllu --gold_file pretagged/pos.hr500k_ud.train+test.conllu --mode predict
# Serbian lemmatization
python -m stanfordnlp.models.lemmatizer --model_dir models/lemma/ --model_file SETimes.SR+srLex --eval_file pretagged/pos.SETimes.SR_ud.dev.conllu --output_file pretagged/pos.lemma.SETimes.SR_ud.dev.conllu --gold_file pretagged/pos.SETimes.SR_ud.dev.conllu --mode predict
python -m stanfordnlp.models.lemmatizer --model_dir models/lemma/ --model_file SETimes.SR+srLex --eval_file pretagged/pos.SETimes.SR_ud.train+test.conllu --output_file pretagged/pos.lemma.SETimes.SR_ud.train+test.conllu --gold_file pretagged/pos.SETimes.SR_ud.train+test.conllu --mode predict
```

Once you have all the training data preprocessed, you can train the parsers in the following manner:

```
python -m stanfordnlp.models.parser --save_dir models/depparse/ --save_name ssj500k_ud --wordvec_file ~/data/clarin.si-embed/embed.sl-token.ft.sg.vec.xz --train_file pretagged/pos.lemma.ssj500k_ud.train+test.conllu --eval_file pretagged/pos.lemma.ssj500k_ud.dev.conllu --gold_file pretagged/pos.lemma.ssj500k_ud.dev.conllu --shorthand sl_ssj --output_file temp.depparse.sl --mode train
python -m stanfordnlp.models.parser --save_dir models/depparse/ --save_name hr500k_ud --wordvec_file ~/data/clarin.si-embed/embed.hr-token.ft.sg.vec.xz --train_file pretagged/pos.lemma.hr500k_ud.train+test.conllu --eval_file pretagged/pos.lemma.hr500k_ud.dev.conllu --gold_file pretagged/pos.lemma.hr500k_ud.dev.conllu --shorthand hr_set --output_file temp.depparse.hr --mode train
python -m stanfordnlp.models.parser --save_dir models/depparse/ --save_name SETimes.SR_ud --wordvec_file ~/data/clarin.si-embed/embed.sr-token.ft.sg.vec.xz --train_file pretagged/pos.lemma.SETimes.SR_ud.train+test.conllu --eval_file pretagged/pos.lemma.SETimes.SR_ud.dev.conllu --gold_file pretagged/pos.lemma.SETimes.SR_ud.dev.conllu --shorthand sr_set --output_file temp.depparse.sr --mode train
```
