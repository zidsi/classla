# A [CLASSLA](http://www.clarin.si/info/k-centre/) Fork of [Stanza](https://github.com/stanfordnlp/stanza) For Processing Slovene, Croatian, Serbian and Bulgarian

## Description

This pipeline allows for processing of Slovene, Croatian, Serbian and Bulgarian on the levels of

- tokenization and sentence splitting
- part-of-speech tagging
- lemmatization
- dependency parsing
- named entity recognition

## Installation
### pip
We recommend that you install CLASSLA via pip, the Python package manager. To install, run:
```bash
pip install classla
```
This will also resolve all dependencies.

## Running CLASSLA
### Getting started
To run the CLASSLA pipeline for the first time, follow these steps:
```
>>> import classla
>>> classla.download('sl')                            # download models for Slovene, use hr for Croatian, sr for Serbian, bg for Bulgarian
>>> nlp = classla.Pipeline('sl')                      # initialize the default Slovene pipeline, use hr for Croatian, sr for Serbian, bg for Bulgarian
>>> doc = nlp("France Prešeren je rojen v Vrbi.")     # run the pipeline
>>> print(doc.conll_file.conll_as_string())           # print the output in CoNLL-U format
# newpar id = 1
# sent_id = 1.1
# text = France Prešeren je rojen v Vrbi.
1	France	France	PROPN	Npmsn	Case=Nom|Gender=Masc|Number=Sing	4	nsubj	_	NER=B-per
2	Prešeren	Prešeren	PROPN	Npmsn	Case=Nom|Gender=Masc|Number=Sing	1	flat_name	_	NER=I-per
3	je	biti	AUX	Va-r3s-n	Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin	4	cop	_	NER=O
4	rojen	rojen	ADJ	Appmsnn	Case=Nom|Definite=Ind|Degree=Pos|Gender=Masc|Number=Sing|VerbForm=Part	0	root	_	NER=O
5	v	v	ADP	Sl	Case=Loc	6	case	_	NER=O
6	Vrbi	Vrba	PROPN	Npfsl	Case=Loc|Gender=Fem|Number=Sing	4	obl	_	NER=B-loc|SpaceAfter=No
7	.	.	PUNCT	Z	_	4	punct	_	NER=O
```

You can also consult the ```pipeline_demo.py``` file for usage examples.

## Processors

The CLASSLA pipeline is built from multiple units. These units are called processors. By default CLASSLA runs the ```tokenize```, ```ner```, ```pos```, ```lemma``` and ```depparse``` processors.

You can specify which processors `CLASSLA should run, via the ```processors``` attribute as in the following example, performing tokenization, named entity recognition, part-of-speech tagging and lemmatization.

```python
>>> nlp = classla.Pipeline('sl', processors='tokenize,ner,pos,lemma')
```

Another popular option might be to perform tokenization, part-of-speech tagging, lemmatization and dependency parsing.

```python
>>> nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma,depparse')
```

### Tokenization and sentence splitting

The tokenization and sentence splitting processor ```tokenize``` is the first processor and is required for any further processing.

In case you already have tokenized text, you should separate tokens via spaces and pass the attribute ```tokenize_pretokenized=True```.

By default CLASSLA uses a rule-based tokenizer - [reldi-tokeniser](https://github.com/clarinsi/reldi-tokeniser).

<!--Most important attributes:
```
tokenize_pretokenized   - [boolean]     ignores tokenizer
```-->

### Part-of-speech tagging

The POS tagging processor ```pos``` will general output that contains morphosyntactic description following the [MULTEXT-East standard](http://nl.ijs.si/ME/V6/msd/html/msd.lang-specific.html) and universal part-of-speech tags and universal features following the [Universal Dependencies standard](https://universaldependencies.org). This processing requires the usage of the ```tokenize``` processor.

<!--Most important attributes:
```
pos_model_path          - [str]         alternative path to model file
pos_pretrain_path       - [str]         alternative path to pretrain file
```-->

### Lemmatization

The lemmatization processor ```lemma``` will produce lemmas (basic forms) for each token in the input. It requires the usage of both the ```tokenize``` and ```pos``` processors.

### Dependency parsing

The dependency parsing processor ```depparse``` performs syntactic dependency parsing of sentences following the [Universal Dependencies formalism](https://universaldependencies.org/introduction.html#:~:text=Universal%20Dependencies%20(UD)%20is%20a,from%20a%20language%20typology%20perspective.). It requires the ```tokenize``` and ```pos``` processors.

### Named entity recognition

The named entity recognition processor ```ner``` identifies named entities in text following the [IOB2](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)) format. It requires only the ```tokenize``` processor.
