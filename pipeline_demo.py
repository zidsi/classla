"""
basic demo script
"""

import sys
import argparse
import os

import classla
from classla.utils.resources import DEFAULT_MODEL_DIR


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--models_dir', help='location of models files | default: ~/classla_resources',
                        default=DEFAULT_MODEL_DIR)
    parser.add_argument('-l', '--lang', help='Demo language',
                        default="sl")
    parser.add_argument('-c', '--cpu', action='store_true', help='Use cpu as the device.')
    args = parser.parse_args()

    example_sentences = {"sl": "France Prešeren je rojen v Vrbi.",
                         "hr": "Ante Starčević rođen je u Velikom Žitniku.",
                         "sr": "Slobodan Jovanović rođen je u Novom Sadu.",
                         "bg": "Алеко Константинов е роден в Свищов."}

    if args.lang not in example_sentences:
        print(f'Sorry, but we don\'t have a demo sentence for "{args.lang}" for the moment. Try one of these languages: {list(example_sentences.keys())}')
        sys.exit(1)

    # download the models
    classla.download(args.lang, args.models_dir, confirm_if_exists=True)
    # set up a pipeline
    print('---')
    print('Building pipeline...')
    pipeline = classla.Pipeline(models_dir=args.models_dir, lang=args.lang, use_gpu=(not args.cpu))
    # process the document
    doc = pipeline(example_sentences[args.lang])
    # access nlp annotations
    print('')
    print('Input: {}'.format(example_sentences[args.lang]))
    print("The tokenizer split the input into {} sentences.".format(len(doc.sentences)))
    print('---')
    print('tokens of first sentence: ')
    doc.sentences[0].print_tokens()
    print('')
    print('---')
    print('dependency parse of first sentence: ')
    doc.sentences[0].print_dependencies()
    print('')
