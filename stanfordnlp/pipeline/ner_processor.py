"""
Processor for performing named entity tagging.
"""

from stanfordnlp.models.common import doc
from stanfordnlp.models.common.utils import unsort
from stanfordnlp.models.ner.data import DataLoader
from stanfordnlp.models.ner.trainer import Trainer
from stanfordnlp.pipeline._constants import *
from stanfordnlp.pipeline.processor import UDProcessor


class NERProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([NER])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, use_gpu):
        # set up trainer
        args = {'charlm_forward_file': config['forward_charlm_path'], 'charlm_backward_file': config['backward_charlm_path']}
        self._trainer = Trainer(args=None, model_file=config['model_path'], use_cuda=use_gpu)

    def process(self, document):
        # set up a eval-only data loader and skip tag preprocessing
        batch = DataLoader(
            document, self.config['batch_size'], self.config, vocab=self.vocab, evaluation=True, preprocess_tags=False)
        preds = []
        for b in batch:
            preds += self.trainer.predict(b)

        # Append previous 'misc' values.
        misc = batch.conll.get(['misc'])
        idx = 0
        for i, sent in enumerate(preds):
            for j, ner_pred in enumerate(sent):
                misc_val = misc[idx]
                if misc_val != '_':
                    preds[i][j] = ner_pred + ',' + misc_val
                idx += 1

        batch.conll.set(['misc'], [y for x in preds for y in x])
        #batch.doc.set([doc.NER], [y for x in preds for y in x], to_token=True)
