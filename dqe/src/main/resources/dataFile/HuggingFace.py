# Load your usual SpaCy model (one of SpaCy English models)
import spacy
import sys
nlp = spacy.load('en_core_web_lg')

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)

# You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.

arg = sys.argv[1]
# print(arg)
# arg = "There is an event Mets Vs Braves at Citi Field today at 7:30 pm. Does it sell Corona beer?"

parts = arg.split('|')
doc = nlp(parts[0])
string = ""

for partss in parts[1:]:
    part = partss.split('$')
    span = doc[int(part[0]):int(part[1])]
    string += str(span._.coref_scores) + "|"

print(string[:-1])

# print(doc._.has_coref)
# print(doc._.coref_clusters)
# print(doc._.coref_scores)

# span = doc[int(sys.argv[2]):int(sys.argv[3])]
# print('span: ' + span)
# print(span._.is_coref)
# print(span._.coref_scores)

