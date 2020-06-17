import stanfordnlp
from stanfordnlp.pipeline.doc import Document
from stanfordnlp.models.common.conll import CoNLLFile


config = {
        'processors': 'tokenize,mwt,ner,pos,lemma,depparse',
        'tokenize_pretokenized': True,
        'pos_model_path': 'models/pos/ssj500k',
        'pos_pretrain_path': 'models/pos/ssj500k.pretrain.pt',
        'pos_batch_size': 1000,
        'lemma_model_path': 'models/lemma/ssj500k+Sloleks_lemmatizer.pt',
        'depparse_model_path': 'models/depparse/ssj500k_ud',
        'depparse_pretrain_path': 'models/pos/ssj500k.pretrain.pt',
        'ner_model_path': 'models/ner/ssj500k',
        'ner_pretrain_path': 'models/ner/ssj500k.pretrain.pt',
        'ner_forward_charlm_path': None, 'ner_backward_charlm_path': None,
        'use_gpu': True
}
nlp = stanfordnlp.Pipeline(**config)

# Because we already have CoNLL-U formated input, we need to skip the tokenization step.
# This is currently done by setting the Documents text parameter as None. After that we also
# have to manually create a CoNLLFile instance and append it to the Document.
doc = Document(text=None)
conll_file = CoNLLFile(filename='data/ssj500k.dev.conllu')
doc.conll_file = conll_file

# Start processing.
res = nlp(doc)

# Save result to output CoNLL-U file.
with open('temp.conllu', 'w') as f:
    f.write(res.conll_file.conll_as_string())
