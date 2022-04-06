"""
Processor for performing named entity tagging.
"""
import logging

from classla.models.common import doc
from classla.models.common.utils import unsort
from classla.models.srl.data import DataLoader
from classla.models.srl.trainer import Trainer
from classla.pipeline._constants import *
from classla.pipeline.processor import UDProcessor, register_processor

from classla.models.common.pretrain import Pretrain

logger = logging.getLogger('classla')


@register_processor(name=SRL)
class SRLProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([SRL])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE, POS, LEMMA, DEPPARSE])

    def _set_up_model(self, config, use_gpu):
        # get pretrained word vectors
        self._pretrain = Pretrain(config['pretrain_path']) if 'pretrain_path' in config else None

        # if 'lemma_pretag' in self.config:
        #     pos_lemma_pretag = self.config['lemma_pretag']
        # else:
        #     pos_lemma_pretag = (not 'tokenize_pretokenized' in self.pipeline.config or not self.pipeline.config[
        #         'tokenize_pretokenized'])
        #     self.config['lemma_pretag'] = pos_lemma_pretag
        #
        # arg = {'lemma_pretag': pos_lemma_pretag}
        arg = {}

        # set up trainer
        self._trainer = Trainer(args=arg, pretrain=self.pretrain, model_file=config['model_path'], use_cuda=use_gpu)

    def predetermined_punctuations(self, seq):
        """ Determine if punctuation is already assigned by tokenizer. """
        return [pos if pos[0] is not None else False for pos in seq]

    def process(self, document):
        batch = DataLoader(
            document, self.config['batch_size'], self.config, self.pretrain, vocab=self.vocab, evaluation=True,
            sort_during_eval=True)
        preds = []
        for i, b in enumerate(batch):
            preds += self.trainer.predict(b)
        preds = unsort(preds, batch.data_orig_idx)
        if 'lemma_pretag' in self.config and self.config['lemma_pretag']:
            preds_flattened = []
            # skip pos predictions for punctuations that were predicted by tokenizer
            skip = iter(self.predetermined_punctuations(zip(batch.doc.get([doc.UPOS]), batch.doc.get([doc.XPOS]), batch.doc.get([doc.FEATS]))))
            for x in preds:
                for y in x:
                    n = next(skip, None)
                    assert n is not None
                    if not n:
                        preds_flattened.append(y)
                    else:
                        preds_flattened.append(n)
        else:
            preds_flattened = [y for x in preds for y in x]

        batch.doc.set([doc.UPOS, doc.XPOS, doc.FEATS], preds_flattened)
        return batch.doc
