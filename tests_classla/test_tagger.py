"""
Basic testing of part of speech tagging
"""

import classla

from tests_classla import *

with open('test_data/slovenian.raw') as f:
    SL_DOC = f.read()

with open('test_data/slovenian.tagger') as f:
    SL_DOC_GOLD = f.read()


def test_part_of_speech():
    nlp = classla.Pipeline(**{'processors': 'tokenize,pos', 'dir': TEST_MODELS_DIR, 'lang': 'sl'})
    doc = nlp(SL_DOC)
    # with open('test_data/slovenian.tagger', 'w') as f:
    #     f.write('\n\n'.join([sent.tokens_string() for sent in doc.sentences]))
    assert SL_DOC_GOLD == '\n\n'.join([sent.tokens_string() for sent in doc.sentences])
