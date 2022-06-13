"""
Basic testing of dependency parser
"""

import classla
from classla.utils.conll import CoNLL

from tests_classla import *

with open('test_data/slovenian.raw') as f:
    SL_DOC = f.read()

with open('test_data/slovenian.srl') as f:
    SL_DOC_GOLD = f.read()


def test_parser():
    nlp = classla.Pipeline(
        **{'processors': 'tokenize,pos,lemma,depparse,srl', 'dir': TEST_MODELS_DIR, 'lang': 'sl', 'type': 'standard_jos'})
    doc = nlp(SL_DOC)
    # with open('test_data/slovenian.srl', 'w') as f:
    #     f.write(doc.to_conll())
    assert SL_DOC_GOLD == doc.to_conll()
