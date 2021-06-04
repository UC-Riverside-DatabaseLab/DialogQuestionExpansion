from flask import Flask
import neuralcoref
import spacy
import sys
import logging

app = Flask(__name__)
@app.route('/')
def hello_world():
    nlp = spacy.load('en_core_web_lg')
    neuralcoref.add_to_pipe(nlp)
    # arg = sys.argv[1]
    arg = "There is an event Mets Vs Braves at Citi Field today at 7:30 pm. Does it sell Corona beer?"
    parts = arg.split('|')
    doc = nlp(parts[0])
    string = ""
    for partss in parts[1:]:
        part = partss.split('$')
        span = doc[int(part[0]):int(part[1])]
        string += str(span._.coref_scores) + "|"
    print(string[:-1]);
    app.logger.info(string[:-1])
    return string[:-1]
if __name__ == '__main__':
    app.run()

