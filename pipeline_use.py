import classla
from classla.pipeline.doc import Document
from classla.models.common.conll import CoNLLFile


config = {
    'processors': 'tokenize,ner,pos,lemma,depparse',
    # 'tokenize_pretokenized': True,
    'lang': 'sr',
    'input_type': 'nonstandard',
    'pos_model_path': 'saved_models/pos/ssj500k',
    'pos_pretrain_path': 'saved_models/pos/ssj500k.pretrain.pt',
    'pos_batch_size': 1000,
    'lemma_model_path': 'saved_models/lemma/ssj500k+Sloleks_lemmatizer.pt',
    'depparse_model_path': 'saved_models/depparse/ssj500k_ud',
    'depparse_pretrain_path': 'saved_models/pos/ssj500k.pretrain.pt',
    'ner_model_path': 'saved_models/ner/ssj500k',
    'ner_pretrain_path': 'saved_models/ner/ssj500k.pretrain.pt',
    'ner_forward_charlm_path': None, 'ner_backward_charlm_path': None,
    'use_gpu': True
}
nlp = classla.Pipeline(**config)

# Because we already have CoNLL-U formated input, we need to skip the tokenization step.
# This is currently done by setting the Documents text parameter as None. After that we also
# have to manually create a CoNLLFile instance and append it to the Document.
doc = Document(text=None)
# conll_file = CoNLLFile(filename='data/ssj500k.dev.conllu')
conll_file = CoNLLFile(filename='data/ssj500k.dev.conllu')
doc.conll_file = conll_file

# Start processing.
with open('data/input.txt', 'r') as f:
    res = nlp(f.read())
    # res = nlp(doc)

# Save result to output CoNLL-U file.
with open('temp.conllu', 'w') as f:
    f.write(res.conll_file.conll_as_string())
