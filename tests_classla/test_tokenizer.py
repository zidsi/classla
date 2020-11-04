"""
Basic testing of tokenization
"""

import classla

from tests import *


SL_DOC = "France Prešeren je bil rojen v Vrbi. Danes je poznan kot največji slovenski pesnik. Študiral je na Dunaju."

SL_DOC_GOLD_TOKENS = """
<Token index=1;words=[<Word index=1;text=France>]>
<Token index=2;words=[<Word index=2;text=Prešeren>]>
<Token index=3;words=[<Word index=3;text=je>]>
<Token index=4;words=[<Word index=4;text=bil>]>
<Token index=5;words=[<Word index=5;text=rojen>]>
<Token index=6;words=[<Word index=6;text=v>]>
<Token index=7;words=[<Word index=7;text=Vrbi>]>
<Token index=8;words=[<Word index=8;text=.>]>

<Token index=1;words=[<Word index=1;text=Danes>]>
<Token index=2;words=[<Word index=2;text=je>]>
<Token index=3;words=[<Word index=3;text=poznan>]>
<Token index=4;words=[<Word index=4;text=kot>]>
<Token index=5;words=[<Word index=5;text=največji>]>
<Token index=6;words=[<Word index=6;text=slovenski>]>
<Token index=7;words=[<Word index=7;text=pesnik>]>
<Token index=8;words=[<Word index=8;text=.>]>

<Token index=1;words=[<Word index=1;text=Študiral>]>
<Token index=2;words=[<Word index=2;text=je>]>
<Token index=3;words=[<Word index=3;text=na>]>
<Token index=4;words=[<Word index=4;text=Dunaju>]>
<Token index=5;words=[<Word index=5;text=.>]>
""".strip()


SL_DOC_PRETOKENIZED = \
    "France Prešeren je bil rojen v Vrbi .\nDanes je poznan kot največji slovenski pesnik .\n\nŠtudiral je na Dunaju.\n"

SL_DOC_PRETOKENIZED_GOLD_TOKENS = """
<Token index=1;words=[<Word index=1;text=France>]>
<Token index=2;words=[<Word index=2;text=Prešeren>]>
<Token index=3;words=[<Word index=3;text=je>]>
<Token index=4;words=[<Word index=4;text=bil>]>
<Token index=5;words=[<Word index=5;text=rojen>]>
<Token index=6;words=[<Word index=6;text=v>]>
<Token index=7;words=[<Word index=7;text=Vrbi>]>
<Token index=8;words=[<Word index=8;text=.>]>

<Token index=1;words=[<Word index=1;text=Danes>]>
<Token index=2;words=[<Word index=2;text=je>]>
<Token index=3;words=[<Word index=3;text=poznan>]>
<Token index=4;words=[<Word index=4;text=kot>]>
<Token index=5;words=[<Word index=5;text=največji>]>
<Token index=6;words=[<Word index=6;text=slovenski>]>
<Token index=7;words=[<Word index=7;text=pesnik>]>
<Token index=8;words=[<Word index=8;text=.>]>

<Token index=1;words=[<Word index=1;text=Študiral>]>
<Token index=2;words=[<Word index=2;text=je>]>
<Token index=3;words=[<Word index=3;text=na>]>
<Token index=4;words=[<Word index=4;text=Dunaju.>]>
""".strip()


SL_DOC_PRETOKENIZED_LIST = [['France', 'Prešeren', 'je', 'bil', 'rojen', 'v', 'Vrbi', '.'], ['Danes', 'živi', 'v',
                                                                                             'poeziji', '.']]

SL_DOC_PRETOKENIZED_LIST_GOLD_TOKENS = """
<Token index=1;words=[<Word index=1;text=France>]>
<Token index=2;words=[<Word index=2;text=Prešeren>]>
<Token index=3;words=[<Word index=3;text=je>]>
<Token index=4;words=[<Word index=4;text=bil>]>
<Token index=5;words=[<Word index=5;text=rojen>]>
<Token index=6;words=[<Word index=6;text=v>]>
<Token index=7;words=[<Word index=7;text=Vrbi>]>
<Token index=8;words=[<Word index=8;text=.>]>

<Token index=1;words=[<Word index=1;text=Danes>]>
<Token index=2;words=[<Word index=2;text=živi>]>
<Token index=3;words=[<Word index=3;text=v>]>
<Token index=4;words=[<Word index=4;text=poeziji>]>
<Token index=5;words=[<Word index=5;text=.>]>
""".strip()


def test_tokenize():
    nlp = classla.Pipeline(processors='tokenize', models_dir=TEST_MODELS_DIR, lang='sl')
    doc = nlp(SL_DOC)
    assert SL_DOC_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])


def test_pretokenized():
    nlp = classla.Pipeline(**{'processors': 'tokenize', 'models_dir': TEST_MODELS_DIR, 'lang': 'sl',
                                  'tokenize_pretokenized': True})
    doc = nlp(SL_DOC_PRETOKENIZED)
    assert SL_DOC_PRETOKENIZED_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
    doc = nlp(SL_DOC_PRETOKENIZED_LIST)
    assert SL_DOC_PRETOKENIZED_LIST_GOLD_TOKENS == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])

