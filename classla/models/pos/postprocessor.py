class InflectionalLexiconProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def process(self, data):
        raise NotImplementedError()


class SloveneInflectionalLexiconProcessor(InflectionalLexiconProcessor):
    def process(self, data):
        return None


processors = {"sj_ssj": SloveneInflectionalLexiconProcessor}


class InflectionalLexicon:
    def __init__(self, lexicon, shorthand):
        """Base class for data converters for sequence classification data sets."""
        self.lexicon = lexicon
        self.shorthand = shorthand
        self.processor = processors[shorthand]

    def process(self, data):
        return self.processor.process(data)
