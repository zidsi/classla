"""
Utilities related to using reldi in the pipeline.
"""
import re


def check_reldi():
    """
    Import necessary components from reldi to perform tokenization.
    """
    try:
        import classla
    except ImportError:
        raise ImportError(
            "Reldi is used but not present on your machine. Make sure you run 'git submodule init' and 'git submodule update' commands."
        )
    return True


class ReldiTrainer():
    def __init__(self, lang='sl', type='standard'):
        """ Construct a reldi-based tokenizer by loading the reldi pipeline.
        """
        if lang not in ['sl', 'hr', 'sr', 'bg', 'mk']:
            raise Exception("Reldi tokenizer is currently only allowed in Slovene, Croatian and Serbian pipelines.")

        check_reldi()
        from classla.submodules.reldi_tokeniser import tokeniser
        self.nlp = tokeniser
        self.lang = lang
        self.type = type
        self.tokenizer = self.nlp.generate_tokenizer(self.lang)
        self.regexes = {
            'punct_re': re.compile(r'^[' + re.escape(tokeniser.punct) + ']+$', re.UNICODE),
            'symb_re': re.compile(r'^[' + re.escape('#%&<>+=°x÷$@µ©™®§') + ']+$'),
            'hashtag_re': re.compile(r'^' + tokeniser.langs[lang]['hashtag'] + '$'),
            'mention_re': re.compile(r'^' + tokeniser.langs[lang]['mention'] + '$'),
            'mail_re': re.compile(r'^' + tokeniser.langs[lang]['mail'] + '$'),
            'url_re': re.compile(r'^' + tokeniser.langs[lang]['url'] + '$'),
            'emoticon_re': re.compile(r'^' + tokeniser.langs[lang]['emoticon'] + '$'),
            'emoji_re': re.compile('^[\U00010000-\U0010ffff]+$', flags=re.UNICODE)
        }

    def _reldi_tokenizer(self, input, par_id):
        """
        Slightly modified function reldi_tokeniser.represent_tomaz. Modifications include:

        1.) Erase of non conllu format options (not relevant for pipeline).

        Original function not used because it uses args, which is not passed as parameter.
        """
        doc = []
        metadoc = []
        sent_id = 0
        metadata = '# newpar id = ' + str(par_id) + '\n'
        for sent_idx, sent in enumerate(input):
            doc_sent = []
            sent_id += 1
            token_id = 0
            metadata += '# sent_id = ' + str(par_id) + '.' + str(sent_id) + '\n'
            metadata += '# text = ' + self.nlp.to_text(sent)
            metadoc.append(metadata)
            metadata = ''
            for token_idx, (token, start, end) in enumerate(sent):
                if not token[0].isspace():
                    tok = {}
                    token_id += 1
                    lemma = None
                    xpos = None
                    upos = None
                    # 'order':('abbrev','num','url','htmlesc','tag','mail','mention','hashtag','emoticon','word','arrow','dot','space','other')
                    if self.regexes['url_re'].match(token):
                        xpos = 'Xw'
                        upos = 'SYM'
                        lemma = token
                    elif self.regexes['mail_re'].match(token):
                        xpos = 'Xw'
                        upos = 'SYM'
                        lemma = token
                    elif self.regexes['mention_re'].match(token):
                        xpos = 'Xa'
                        upos = 'SYM'
                        lemma = token
                    elif self.regexes['hashtag_re'].match(token):
                        xpos = 'Xh'
                        upos = 'SYM'
                        lemma = token
                    elif self.regexes['emoticon_re'].match(token):
                        xpos = 'Xe'
                        upos = 'SYM'
                        lemma = token
                    elif self.regexes['emoji_re'].match(token):
                        xpos = 'Xe'
                        upos = 'SYM'
                        lemma = token
                    elif self.regexes['punct_re'].match(token):
                        xpos = 'Z'
                        lemma = token
                        if self.regexes['symb_re'].match(token):
                            upos = 'SYM'
                        else:
                            upos = 'PUNCT'
                    SpaceAfter = True
                    if len(sent) > token_idx + 1:
                        SpaceAfter = sent[token_idx + 1][0].isspace()
                    elif len(input) > sent_idx + 1:
                        SpaceAfter = input[sent_idx + 1][0][0].isspace()
                    tok['id'] = tuple([token_id])
                    tok['text'] = token
                    if lemma:
                        tok['lemma'] = lemma
                    if xpos:
                        tok['xpos'] = xpos
                    if upos:
                        tok['upos'] = upos
                    if SpaceAfter:
                        pass
                    else:
                        tok['misc'] = 'SpaceAfter=No'
                    doc_sent.append(tok)
            doc.append(doc_sent)
        return doc, metadoc


    def tokenize(self, document):
        """ Tokenize a document with the reldi tokenizer and add results to document.conll_file.
        """
        raw_text = '\n'.join(document) if isinstance(document, list) else document
        document = []
        metadocument = []
        par_id = 1
        for text in raw_text.split('\n'):
            if text == '' or text.isspace():
                continue
            doc, metadoc = self._reldi_tokenizer(self.nlp.process[self.type](self.tokenizer, text, self.lang), par_id)
            document += doc
            metadocument += metadoc
            par_id += 1

        return raw_text, document, metadocument
