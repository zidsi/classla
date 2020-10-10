import torch


class InflectionalLexiconProcessor(object):
    def __init__(self, lexicon, vocab, pretrain):
        pass

    """Base class for data converters for sequence classification data sets."""
    def process(self, padded_prediction, word_ids):
        raise NotImplementedError()


class SloveneInflectionalLexiconProcessor(InflectionalLexiconProcessor):
    def __init__(self, lexicon, vocab, pretrain):
        closed_classes_rules = ['P', 'S', 'C', 'Q']

        # fills hypothesis_dictionary
        self.hypothesis_dictionary = {}
        self.extract_lexicon_data(lexicon, vocab['xpos'], pretrain.vocab)

        # fills closed_classes
        self.closed_classes = set()
        self.create_closed_classes(vocab['xpos'], closed_classes_rules)
        super(SloveneInflectionalLexiconProcessor, self).__init__(lexicon, vocab, pretrain)

    def process(self, padded_prediction, word_ids):
        predictions = []

        sorted, sorted_indices = torch.sort(padded_prediction, 2, True)

        for sent_indices, sent_ids in zip(sorted_indices, word_ids):
            sent_predictions = []
            for word_indices, word_id in zip(sent_indices, sent_ids):
                if int(word_id) in self.hypothesis_dictionary:
                    for ind in word_indices:
                        if int(ind) in self.hypothesis_dictionary[int(word_id)]:
                            prediction = ind
                            break
                else:
                    # in case word is a legit member of closed classes it should already be handled by hypothesis dict
                    for ind in word_indices:
                        if int(ind) not in self.closed_classes:
                            prediction = ind
                            break
                sent_predictions.append(prediction)
            predictions.append(torch.stack(sent_predictions))
        return torch.stack(predictions)

    def extract_lexicon_data(self, lexicon, vocab_xpos, vocab_words):
        """ Creates hypothesis dictionary from lexicon. """
        for key in lexicon.keys():
            if key[1] in vocab_xpos and key[0] in vocab_words:
                self.hypothesis_dictionary.setdefault(vocab_words[key[0]], []).append(vocab_xpos[key[1]])

    def create_closed_classes(self, vocab, closed_classes_rules):
        """ Fills a set of closed classes, that contains xpos ids that are not permitted. """
        for key in vocab:
            if key[0] in closed_classes_rules:
                self.closed_classes.add(vocab[key])


processors = {"sl_ssj": SloveneInflectionalLexiconProcessor}


class InflectionalLexicon:
    def __init__(self, lexicon, shorthand, vocab, pretrain):
        """Base class for data converters for sequence classification data sets."""
        self.shorthand = shorthand
        assert shorthand in processors, f"Tag {shorthand} is not supported by inflectional lexicon."
        self.processor = processors[shorthand](lexicon, vocab, pretrain)

    def process(self, padded_prediction, word_ids):
        return self.processor.process(padded_prediction, word_ids)
