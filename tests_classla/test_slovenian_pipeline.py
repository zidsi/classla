"""
Basic testing of the English pipeline
"""

import pytest
import classla

from tests import *


# data for testing
SL_DOC = "France Prešeren je bil rojen v Vrbi. Danes je poznan kot največji slovenski pesnik. Študiral je na Dunaju."

SL_DOC_TOKENS_GOLD = """
<Token index=1;words=[<Word index=1;text=France;lemma=France;upos=PROPN;xpos=Npmsn;feats=Case=Nom|Gender=Masc|Number=Sing;governor=5;dependency_relation=nsubj>]>
<Token index=2;words=[<Word index=2;text=Prešeren;lemma=Prešeren;upos=PROPN;xpos=Npmsn;feats=Case=Nom|Gender=Masc|Number=Sing;governor=1;dependency_relation=flat_name>]>
<Token index=3;words=[<Word index=3;text=je;lemma=biti;upos=AUX;xpos=Va-r3s-n;feats=Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin;governor=5;dependency_relation=aux>]>
<Token index=4;words=[<Word index=4;text=bil;lemma=biti;upos=AUX;xpos=Va-p-sm;feats=Gender=Masc|Number=Sing|VerbForm=Part;governor=5;dependency_relation=cop>]>
<Token index=5;words=[<Word index=5;text=rojen;lemma=rojen;upos=ADJ;xpos=Appmsnn;feats=Case=Nom|Definite=Ind|Degree=Pos|Gender=Masc|Number=Sing|VerbForm=Part;governor=0;dependency_relation=root>]>
<Token index=6;words=[<Word index=6;text=v;lemma=v;upos=ADP;xpos=Sl;feats=Case=Loc;governor=7;dependency_relation=case>]>
<Token index=7;words=[<Word index=7;text=Vrbi;lemma=Vrba;upos=PROPN;xpos=Npfsl;feats=Case=Loc|Gender=Fem|Number=Sing;governor=5;dependency_relation=obl>]>
<Token index=8;words=[<Word index=8;text=.;lemma=.;upos=PUNCT;xpos=Z;feats=_;governor=5;dependency_relation=punct>]>

<Token index=1;words=[<Word index=1;text=Danes;lemma=danes;upos=ADV;xpos=Rgp;feats=Degree=Pos;governor=3;dependency_relation=advmod>]>
<Token index=2;words=[<Word index=2;text=je;lemma=biti;upos=AUX;xpos=Va-r3s-n;feats=Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin;governor=3;dependency_relation=cop>]>
<Token index=3;words=[<Word index=3;text=poznan;lemma=poznan;upos=ADJ;xpos=Appmsnn;feats=Case=Nom|Definite=Ind|Degree=Pos|Gender=Masc|Number=Sing|VerbForm=Part;governor=0;dependency_relation=root>]>
<Token index=4;words=[<Word index=4;text=kot;lemma=kot;upos=SCONJ;xpos=Cs;feats=_;governor=7;dependency_relation=case>]>
<Token index=5;words=[<Word index=5;text=največji;lemma=velik;upos=ADJ;xpos=Agsmsny;feats=Case=Nom|Definite=Def|Degree=Sup|Gender=Masc|Number=Sing;governor=7;dependency_relation=amod>]>
<Token index=6;words=[<Word index=6;text=slovenski;lemma=slovenski;upos=ADJ;xpos=Agpmsny;feats=Case=Nom|Definite=Def|Degree=Pos|Gender=Masc|Number=Sing;governor=7;dependency_relation=amod>]>
<Token index=7;words=[<Word index=7;text=pesnik;lemma=pesnik;upos=NOUN;xpos=Ncmsn;feats=Case=Nom|Gender=Masc|Number=Sing;governor=3;dependency_relation=obl>]>
<Token index=8;words=[<Word index=8;text=.;lemma=.;upos=PUNCT;xpos=Z;feats=_;governor=3;dependency_relation=punct>]>

<Token index=1;words=[<Word index=1;text=Študiral;lemma=študirati;upos=VERB;xpos=Vmpp-sm;feats=Aspect=Imp|Gender=Masc|Number=Sing|VerbForm=Part;governor=0;dependency_relation=root>]>
<Token index=2;words=[<Word index=2;text=je;lemma=biti;upos=AUX;xpos=Va-r3s-n;feats=Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin;governor=1;dependency_relation=aux>]>
<Token index=3;words=[<Word index=3;text=na;lemma=na;upos=ADP;xpos=Sl;feats=Case=Loc;governor=4;dependency_relation=case>]>
<Token index=4;words=[<Word index=4;text=Dunaju;lemma=Dunaj;upos=PROPN;xpos=Npmsl;feats=Case=Loc|Gender=Masc|Number=Sing;governor=1;dependency_relation=obl>]>
<Token index=5;words=[<Word index=5;text=.;lemma=.;upos=PUNCT;xpos=Z;feats=_;governor=1;dependency_relation=punct>]>
""".strip()

SL_DOC_WORDS_GOLD = """
<Word index=1;text=France;lemma=France;upos=PROPN;xpos=Npmsn;feats=Case=Nom|Gender=Masc|Number=Sing;governor=5;dependency_relation=nsubj>
<Word index=2;text=Prešeren;lemma=Prešeren;upos=PROPN;xpos=Npmsn;feats=Case=Nom|Gender=Masc|Number=Sing;governor=1;dependency_relation=flat_name>
<Word index=3;text=je;lemma=biti;upos=AUX;xpos=Va-r3s-n;feats=Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin;governor=5;dependency_relation=aux>
<Word index=4;text=bil;lemma=biti;upos=AUX;xpos=Va-p-sm;feats=Gender=Masc|Number=Sing|VerbForm=Part;governor=5;dependency_relation=cop>
<Word index=5;text=rojen;lemma=rojen;upos=ADJ;xpos=Appmsnn;feats=Case=Nom|Definite=Ind|Degree=Pos|Gender=Masc|Number=Sing|VerbForm=Part;governor=0;dependency_relation=root>
<Word index=6;text=v;lemma=v;upos=ADP;xpos=Sl;feats=Case=Loc;governor=7;dependency_relation=case>
<Word index=7;text=Vrbi;lemma=Vrba;upos=PROPN;xpos=Npfsl;feats=Case=Loc|Gender=Fem|Number=Sing;governor=5;dependency_relation=obl>
<Word index=8;text=.;lemma=.;upos=PUNCT;xpos=Z;feats=_;governor=5;dependency_relation=punct>

<Word index=1;text=Danes;lemma=danes;upos=ADV;xpos=Rgp;feats=Degree=Pos;governor=3;dependency_relation=advmod>
<Word index=2;text=je;lemma=biti;upos=AUX;xpos=Va-r3s-n;feats=Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin;governor=3;dependency_relation=cop>
<Word index=3;text=poznan;lemma=poznan;upos=ADJ;xpos=Appmsnn;feats=Case=Nom|Definite=Ind|Degree=Pos|Gender=Masc|Number=Sing|VerbForm=Part;governor=0;dependency_relation=root>
<Word index=4;text=kot;lemma=kot;upos=SCONJ;xpos=Cs;feats=_;governor=7;dependency_relation=case>
<Word index=5;text=največji;lemma=velik;upos=ADJ;xpos=Agsmsny;feats=Case=Nom|Definite=Def|Degree=Sup|Gender=Masc|Number=Sing;governor=7;dependency_relation=amod>
<Word index=6;text=slovenski;lemma=slovenski;upos=ADJ;xpos=Agpmsny;feats=Case=Nom|Definite=Def|Degree=Pos|Gender=Masc|Number=Sing;governor=7;dependency_relation=amod>
<Word index=7;text=pesnik;lemma=pesnik;upos=NOUN;xpos=Ncmsn;feats=Case=Nom|Gender=Masc|Number=Sing;governor=3;dependency_relation=obl>
<Word index=8;text=.;lemma=.;upos=PUNCT;xpos=Z;feats=_;governor=3;dependency_relation=punct>

<Word index=1;text=Študiral;lemma=študirati;upos=VERB;xpos=Vmpp-sm;feats=Aspect=Imp|Gender=Masc|Number=Sing|VerbForm=Part;governor=0;dependency_relation=root>
<Word index=2;text=je;lemma=biti;upos=AUX;xpos=Va-r3s-n;feats=Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin;governor=1;dependency_relation=aux>
<Word index=3;text=na;lemma=na;upos=ADP;xpos=Sl;feats=Case=Loc;governor=4;dependency_relation=case>
<Word index=4;text=Dunaju;lemma=Dunaj;upos=PROPN;xpos=Npmsl;feats=Case=Loc|Gender=Masc|Number=Sing;governor=1;dependency_relation=obl>
<Word index=5;text=.;lemma=.;upos=PUNCT;xpos=Z;feats=_;governor=1;dependency_relation=punct>
""".strip()

SL_DOC_DEPENDENCY_PARSES_GOLD = """
('France', '5', 'nsubj')
('Prešeren', '1', 'flat_name')
('je', '5', 'aux')
('bil', '5', 'cop')
('rojen', '0', 'root')
('v', '7', 'case')
('Vrbi', '5', 'obl')
('.', '5', 'punct')

('Danes', '3', 'advmod')
('je', '3', 'cop')
('poznan', '0', 'root')
('kot', '7', 'case')
('največji', '7', 'amod')
('slovenski', '7', 'amod')
('pesnik', '3', 'obl')
('.', '3', 'punct')

('Študiral', '0', 'root')
('je', '1', 'aux')
('na', '4', 'case')
('Dunaju', '1', 'obl')
('.', '1', 'punct')
""".strip()

SL_DOC_CONLLU_GOLD = """
# newpar id = 1
# sent_id = 1.1
# text = France Prešeren je bil rojen v Vrbi.
1	France	France	PROPN	Npmsn	Case=Nom|Gender=Masc|Number=Sing	5	nsubj	_	NER=B-PER
2	Prešeren	Prešeren	PROPN	Npmsn	Case=Nom|Gender=Masc|Number=Sing	1	flat_name	_	NER=I-PER
3	je	biti	AUX	Va-r3s-n	Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin	5	aux	_	NER=O
4	bil	biti	AUX	Va-p-sm	Gender=Masc|Number=Sing|VerbForm=Part	5	cop	_	NER=O
5	rojen	rojen	ADJ	Appmsnn	Case=Nom|Definite=Ind|Degree=Pos|Gender=Masc|Number=Sing|VerbForm=Part	0	root	_	NER=O
6	v	v	ADP	Sl	Case=Loc	7	case	_	NER=O
7	Vrbi	Vrba	PROPN	Npfsl	Case=Loc|Gender=Fem|Number=Sing	5	obl	_	NER=B-LOC|SpaceAfter=No
8	.	.	PUNCT	Z	_	5	punct	_	NER=O

# sent_id = 1.2
# text = Danes je poznan kot največji slovenski pesnik.
1	Danes	danes	ADV	Rgp	Degree=Pos	3	advmod	_	NER=O
2	je	biti	AUX	Va-r3s-n	Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin	3	cop	_	NER=O
3	poznan	poznan	ADJ	Appmsnn	Case=Nom|Definite=Ind|Degree=Pos|Gender=Masc|Number=Sing|VerbForm=Part	0	root	_	NER=O
4	kot	kot	SCONJ	Cs	_	7	case	_	NER=O
5	največji	velik	ADJ	Agsmsny	Case=Nom|Definite=Def|Degree=Sup|Gender=Masc|Number=Sing	7	amod	_	NER=O
6	slovenski	slovenski	ADJ	Agpmsny	Case=Nom|Definite=Def|Degree=Pos|Gender=Masc|Number=Sing	7	amod	_	NER=O
7	pesnik	pesnik	NOUN	Ncmsn	Case=Nom|Gender=Masc|Number=Sing	3	obl	_	NER=O|SpaceAfter=No
8	.	.	PUNCT	Z	_	3	punct	_	NER=O

# sent_id = 1.3
# text = Študiral je na Dunaju.
1	Študiral	študirati	VERB	Vmpp-sm	Aspect=Imp|Gender=Masc|Number=Sing|VerbForm=Part	0	root	_	NER=O
2	je	biti	AUX	Va-r3s-n	Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin	1	aux	_	NER=O
3	na	na	ADP	Sl	Case=Loc	4	case	_	NER=O
4	Dunaju	Dunaj	PROPN	Npmsl	Case=Loc|Gender=Masc|Number=Sing	1	obl	_	NER=B-LOC|SpaceAfter=No
5	.	.	PUNCT	Z	_	1	punct	_	NER=O

""".lstrip()


@pytest.fixture(scope="module")
def processed_doc():
    """ Document created by running full Slovenian pipeline on a few sentences """
    nlp = classla.Pipeline(models_dir=TEST_MODELS_DIR)
    return nlp(SL_DOC)


def test_text(processed_doc):
    assert processed_doc.text == SL_DOC


def test_conllu(processed_doc):
    assert processed_doc.conll_file.conll_as_string() == SL_DOC_CONLLU_GOLD


def test_tokens(processed_doc):
    assert "\n\n".join([sent.tokens_string() for sent in processed_doc.sentences]) == SL_DOC_TOKENS_GOLD


def test_words(processed_doc):
    assert "\n\n".join([sent.words_string() for sent in processed_doc.sentences]) == SL_DOC_WORDS_GOLD


def test_dependency_parse(processed_doc):
    assert "\n\n".join([sent.dependencies_string() for sent in processed_doc.sentences]) == \
           SL_DOC_DEPENDENCY_PARSES_GOLD
