"""
Basic testing of part of speech tagging
"""

import classla
from classla.utils.conll import CoNLL

from tests_classla import *

with open('test_data/slovenian.raw') as f:
    SL_DOC = f.read()

with open('test_data/slovenian.ner') as f:
    SL_DOC_GOLD = f.read()


def test_ner():
    nlp = classla.Pipeline(**{'processors': 'tokenize,ner', 'dir': TEST_MODELS_DIR, 'lang': 'sl'})
    doc = nlp(SL_DOC)
    # with open('test_data/slovenian.ner', 'w') as f:
    #     f.write(doc.to_conll())
    assert SL_DOC_GOLD == doc.to_conll()
