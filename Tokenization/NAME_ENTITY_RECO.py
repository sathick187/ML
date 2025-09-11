from nltk.tokenize import sent_tokenize
sentence="The Eiffel Tower was built from 1887 to 1889 by French engineer Gustave Eiffel, whose company specialized in building metal frameworks and structures."
"""
Person Eg: Krish C Naik
Place Or Location Eg: India
Date Eg: September,24-09-1989
Time  Eg: 4:30pm
Money Eg: 1 million dollar
Organization Eg: iNeuron Private Limited
Percent Eg: 20%, twenty percent
"""
import nltk
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker_tab')
# nltk.download('words')
sent=sent_tokenize(sentence)
from nltk.tokenize import word_tokenize
word=word_tokenize(sentence)
# print(word)
import nltk
tag=nltk.pos_tag(word)
# print(tag)
ner=nltk.ne_chunk(tag).draw()
print(ner)