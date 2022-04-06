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

logger = logging.getLogger('classla')

# TODO!!!
@register_processor(name=SRL)
class SRLProcessor(UDProcessor):

    # set of processor requirements this processor fulfills
    PROVIDES_DEFAULT = set([SRL])
    # set of processor requirements for this processor
    REQUIRES_DEFAULT = set([TOKENIZE])

    def _set_up_model(self, config, use_gpu):
        # set up trainer
        args = {'charlm_forward_file': config['forward_charlm_path'], 'charlm_backward_file': config['backward_charlm_path']}
        self._trainer = Trainer(args=args, model_file=config['model_path'], use_cuda=use_gpu)

    def process(self, document):
        # set up a eval-only data loader and skip tag preprocessing
        batch = DataLoader(
            document, self.config['batch_size'], self.config, vocab=self.vocab, evaluation=True, preprocess_tags=False)
        preds = []
        for i, b in enumerate(batch):
            preds += self.trainer.predict(b)
        batch.doc.set([doc.SRL], [y for x in preds for y in x], to_token=True)
        # collect entities into document attribute
        total = len(batch.doc.build_ents())
        logger.debug(f'{total} entities found in document.')
        return batch.doc
