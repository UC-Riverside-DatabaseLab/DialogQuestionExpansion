# DialogQuestionExpansion

## Before you run
- Put "glove.6B.50d.txt" file in `src/main/resources/dataFile/`. Download [link](http://nlp.stanford.edu/data/glove.6B.zip)
- Install Python env
	1. Install python `version=3.7.*`. [Link](https://www.python.org/downloads/)
	2. Install SpaCy. [Link](https://spacy.io/usage)
		```
		$ pip install -U pip setuptools wheel
		$ pip install -U spacy
		$ python -m spacy download en_core_web_lg
		```
	3. Install NeuralCoref. [Link](https://github.com/huggingface/neuralcoref)
		```
		$ pip install neuralcoref
		```
	