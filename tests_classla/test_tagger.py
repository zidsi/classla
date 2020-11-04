"""
Basic testing of part of speech tagging
"""

import classla

from tests import *

SL_DOC = "France Prešeren se je rodil v vrbi."


SL_DOC_GOLD = """
<Token index=1;words=[<Word index=1;text=France;upos=PROPN;xpos=Npmsn;feats=Case=Nom|Gender=Masc|Number=Sing>]>
<Token index=2;words=[<Word index=2;text=Prešeren;upos=PROPN;xpos=Npmsn;feats=Case=Nom|Gender=Masc|Number=Sing>]>
<Token index=3;words=[<Word index=3;text=se;upos=PRON;xpos=Px------y;feats=PronType=Prs|Reflex=Yes|Variant=Short>]>
<Token index=4;words=[<Word index=4;text=je;upos=AUX;xpos=Va-r3s-n;feats=Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin>]>
<Token index=5;words=[<Word index=5;text=rodil;upos=VERB;xpos=Vmbp-sm;feats=Gender=Masc|Number=Sing|VerbForm=Part>]>
<Token index=6;words=[<Word index=6;text=v;upos=ADP;xpos=Sl;feats=Case=Loc>]>
<Token index=7;words=[<Word index=7;text=vrbi;upos=NOUN;xpos=Ncfsl;feats=Case=Loc|Gender=Fem|Number=Sing>]>
<Token index=8;words=[<Word index=8;text=.;upos=PUNCT;xpos=Z;feats=_>]>
""".strip()


def test_part_of_speech():
    nlp = classla.Pipeline(**{'processors': 'tokenize,pos', 'models_dir': TEST_MODELS_DIR, 'lang': 'sl'})
    doc = nlp(SL_DOC)
    assert SL_DOC_GOLD == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
