# A [CLASSLA](http://www.clarin.si/info/k-centre/) Fork of [Stanza](https://github.com/stanfordnlp/stanza) For Processing South Slavic Languages 
## Installation
### pip
We recommend that you install Classla via pip, the Python package manager. To install, run:
```bash
pip install classla
```
This will also help to resolve all dependencies.

## Running Classla
### Getting started
To run your first Classla pipeline, follow these steps:
```python
>>> import classla
>>> classla.download('sl')                            # to download models in Slovene
>>> nlp = classla.Pipeline('sl')                      # to initialize default Slovene pipeline
>>> doc = nlp("France Prešeren je rojen v Vrbi.")     # to run pipeline
>>> print(doc.conll_file.conll_as_string())           # to print output in conllu format
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

You can also look into ```pipeline_demo.py``` file for usage examples.

## Processors

Classla pipeline is built from multiple units. These units are called processors. By default classla runs tokenize, ner, pos, lemma and depparse processors.

You can specify which processors classla runs, with ```processors``` attribute as in the following example.

```python
>>> nlp = classla.Pipeline('sl', processors='tokenize,ner,pos,lemma')
```

### Tokenization (tokenize)

In case you already have tokenized text, you should split the text (with i.e. spaces) and pass attribute ```tokenize_pretokenized=True```.

By default classla uses a rule-based tokenizer - [reldi-tokeniser](https://github.com/clarinsi/reldi-tokeniser).

Most important attributes:
```
tokenize_pretokenized   - [boolean]     ignores tokenizer
```

### Part-of-speech tagging (pos)

Pos tagging processor will create output, that will contain part-of-speech tags and other features presented on [universal dependencies webiste](https://universaldependencies.org/u/feat/index.html) . It is optional and requires you to use tokenize processor beforehand.

<!--Most important attributes:
```
pos_model_path          - [str]         alternative path to model file
pos_pretrain_path       - [str]         alternative path to pretrain file
```-->

### Lemmatisation (lemma)

Lemmatization processor will produce lemmas for each word in input. It requires the usage of both tokenize and pos processors.

### Parsing (depparse)

Parsing processor (named ```depparse``` in code) creates connections between words explained on [universal dependencies website](https://universaldependencies.org/introduction.html#:~:text=Universal%20Dependencies%20(UD)%20is%20a,from%20a%20language%20typology%20perspective.) . It requires tokenizer and pos processors.

### NER (ner)

Ner processor will try to find named entities in text. It requires tokenize processor.
