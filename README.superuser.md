# Superuser instructions

Super user instructions are dedicated to people, who would like to use additional features that produce different output or enable some additional feature.

## JOS dependency parsing system 

In Slovenian it is possible to replace UD dependency parsing system with JOS parsing system.

### Example of JOS dependency parsing for Slovenian 
```
>>> import classla
>>> classla.download('sl', type='standard_jos')                            # download standard models for Slovene, use hr for Croatian, sr for Serbian, bg for Bulgarian, mk for Macedonian
>>> nlp = classla.Pipeline('sl', type='standard_jos')                      # initialize the default Slovene pipeline, use hr for Croatian, sr for Serbian, bg for Bulgarian, mk for Macedonian
>>> doc = nlp("France Prešeren je rojen v Vrbi.")     # run the pipeline
>>> print(doc.to_conll())                             # print the output in CoNLL-U format
# newpar id = 1
# sent_id = 1.1
# text = France Prešeren je rojen v Vrbi.
1	France	France	PROPN	Npmsn	Case=Nom|Gender=Masc|Number=Sing	3	Sb	_	NER=B-PER
2	Prešeren	Prešeren	PROPN	Npmsn	Case=Nom|Gender=Masc|Number=Sing	1	Atr	_	NER=I-PER
3	je	biti	AUX	Va-r3s-n	Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin	0	Root	_	NER=O
4	rojen	rojen	ADJ	Appmsnn	Case=Nom|Definite=Ind|Degree=Pos|Gender=Masc|Number=Sing|VerbForm=Part	3	Atr	_	NER=O
5	v	v	ADP	Sl	Case=Loc	6	Atr	_	NER=O
6	Vrbi	Vrba	PROPN	Npfsl	Case=Loc|Gender=Fem|Number=Sing	3	AdvO	_	NER=B-LOC|SpaceAfter=No
7	.	.	PUNCT	Z	_	0	Root	_	NER=O

```

## Usage of inflectional lexicon

Slovenian standard morphosyntactic tagging model also supports usage of an inflectional lexicon. To use it, `pos_use_lexicon` must be set to `True`.

The inflectional lexicon usage limits the tags appied to a token covered by the lexicon to conform to the hypotheses in the lexicon. Furthermore, it disallows for any token to be a closed-class token, except for those defined like that in the inflectional lexicon.

### Example of ```pos_use_lexicon``` usage
```
>>> import classla
>>> nlp = classla.Pipeline('sl', pos_use_lexicon=True)
>>> doc = nlp("France Prešeren je rojen v Vrbi.")       # run the pipeline
>>> print(doc.to_conll())                               # print the output in CoNLL-U format
# newpar id = 1
# sent_id = 1.1
# text = France Prešeren je rojen v Vrbi.
1	France	France	PROPN	Npmsn	Case=Nom|Gender=Masc|Number=Sing	4	nsubj	_	NER=B-PER
2	Prešeren	Prešeren	PROPN	Npmsn	Case=Nom|Gender=Masc|Number=Sing	1	flat:name	_	NER=I-PER
3	je	biti	AUX	Va-r3s-n	Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin	4	cop	_	NER=O
4	rojen	rojen	ADJ	Appmsnn	Case=Nom|Definite=Ind|Degree=Pos|Gender=Masc|Number=Sing|VerbForm=Part	0	root	_	NER=O
5	v	v	ADP	Sl	Case=Loc	6	case	_	NER=O
6	Vrbi	Vrba	PROPN	Npfsl	Case=Loc|Gender=Fem|Number=Sing	4	obl	_	NER=B-LOC|SpaceAfter=No
7	.	.	PUNCT	Z	_	4	punct	_	NER=O

```

## Usage of tagging control via the tokenizer

Tagging control via the tokenizer is used by default, unless you are using pretokenized text. The tokenizer gives LEMMA, XPOS, UPOS and FEATS information for phenomena such as punctuation, symbols, e-mails, URLs, mentions, hashtags, emoticons and emojis. You may override default setting by using the boolean `pos_tagging_control` attribute.

### Example of `pos_tagging_control` usage
```
>>> import classla
>>> nlp = classla.Pipeline('sl', pos_tagging_control=False)  # initialize the default Slovene pipeline
>>> doc = nlp("France Prešeren je rojen v Vrbi.")       # run the pipeline
>>> print(doc.to_conll())                               # print the output in CoNLL-U format
# newpar id = 1
# sent_id = 1.1
# text = France Prešeren je rojen v Vrbi.
1	France	France	PROPN	Npmsn	Case=Nom|Gender=Masc|Number=Sing	4	nsubj	_	NER=B-PER
2	Prešeren	Prešeren	PROPN	Npmsn	Case=Nom|Gender=Masc|Number=Sing	1	flat:name	_	NER=I-PER
3	je	biti	AUX	Va-r3s-n	Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin	4	cop	_	NER=O
4	rojen	rojen	ADJ	Appmsnn	Case=Nom|Definite=Ind|Degree=Pos|Gender=Masc|Number=Sing|VerbForm=Part	0	root	_	NER=O
5	v	v	ADP	Sl	Case=Loc	6	case	_	NER=O
6	Vrbi	Vrba	PROPN	Npfsl	Case=Loc|Gender=Fem|Number=Sing	4	obl	_	NER=B-LOC|SpaceAfter=No
7	.	.	PUNCT	Z	_	4	punct	_	NER=O

```

## Pretokenized data

In addition to ```tokenize_pretokenized=True```, which expects space-separated tokens, you can set this attribute to ```tokenize_pretokenized='conllu'```. With this, you may pass pretokenized input string in conllu format (make sure it is formatted properly).

### Example of ```tokenize_pretokenized='conllu'```
```
>>> import classla
>>> nlp = classla.Pipeline('sl', tokenize_pretokenized='conllu')
>>> conllu_pretokenized = """
# newpar id = 1
# sent_id = 1.1
# text = France Prešeren je rojen v Vrbi.
1	France	France	_	_	_	_	_	_	_
2	Prešeren	Prešeren	_	_	_	_	_	_	_
3	je	biti	_	_	_	_	_	_	_
4	rojen	rojen	_	_	_	_	_	_	_
5	v	v	_	_	_	_	_	_	_
6	Vrbi	Vrba	_	_	_	_	_	_	SpaceAfter=No
7	.	.	_	_	_	_	_	_	_

"""
>>> doc = nlp(conllu_pretokenized)
>>> print(doc.to_conll())
# newpar id = 1
# sent_id = 1.1
# text = France Prešeren je rojen v Vrbi.
1	France	France	PROPN	Npmsn	Case=Nom|Gender=Masc|Number=Sing	4	nsubj	_	NER=B-PER
2	Prešeren	Prešeren	PROPN	Npmsn	Case=Nom|Gender=Masc|Number=Sing	1	flat:name	_	NER=I-PER
3	je	biti	AUX	Va-r3s-n	Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin	4	cop	_	NER=O
4	rojen	rojen	ADJ	Appmsnn	Case=Nom|Definite=Ind|Degree=Pos|Gender=Masc|Number=Sing|VerbForm=Part	0	root	_	NER=O
5	v	v	ADP	Sl	Case=Loc	6	case	_	NER=O
6	Vrbi	Vrba	PROPN	Npfsl	Case=Loc|Gender=Fem|Number=Sing	4	obl	_	NER=B-LOC|SpaceAfter=No
7	.	.	PUNCT	Z	_	4	punct	_	NER=O

```
