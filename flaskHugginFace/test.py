# Load your usual SpaCy model (one of SpaCy English models)
import spacy
import sys
nlp = spacy.load('en_core_web_lg')

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)

# You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.

# arg = sys.argv[1]
# print(arg)
arg = "There is an event Mets Vs Braves at Citi Field today at 7:30 pm. Does it sell Corona beer?"
print(arg)

parts = arg.split('|')
print("printing parts [1:]")
print(parts)
print(parts[0])
doc = nlp(parts[0])
print("printing doc")
print(doc)
string = ""
print(doc._.coref_scores)

for partss in parts[1:]:
    print(partss)
    part = partss.split('$')
    print(part)
    span = doc[int(part[0]):int(part[1])]

    string += str(span._.coref_scores) + "|"
print(string[:-1])


# import spacy
# import neuralcoref
# nlp = spacy.load('en_core_web_lg')
# neuralcoref.add_to_pipe(nlp)
# doc = nlp(u'Phone area code will be valid only when all the below conditions are met. It cannot be left blank. It should be numeric. It cannot be less than 200. Minimum number of digits should be 3.')
# # There is an event Mets Vs Braves at Citi Field today at 7:30 pm. Does it sell Corona beer?
# print(doc)
# print(doc._.has_coref)
# print(doc._.coref_clusters)