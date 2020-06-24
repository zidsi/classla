import classla

# Download model if not present
classla.download('sl')
# classla.download('sl', resource_dir='/home/luka/classla_resources')

# Initialize pipeline
# nlp = classla.Pipeline('sl') # parameter type='nonstandard' to be added through the second task
nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma') # if we do not need the full pipeline

# Start processing string.
doc = nlp("Janez Janša je rojen v Grosuplju.") # run annotation over a sentence

# Start processing text file.
# with open('data/input.txt', 'r') as f:
#     doc = nlp(f.read())

# Save result to output CoNLL-U file.
with open('temp.conllu', 'w') as f:
    f.write(doc.conll_file.conll_as_string())
