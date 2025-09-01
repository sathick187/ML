## PARAGRAPH TO Sentance

a="Hello Sathick's, How are you. Where are you going today! every thing is alright."

from nltk.tokenize import sent_tokenize
# import nltk
# nltk.download('punkt_tab')
# nltk.download('punkt')     # only one time

corpus=sent_tokenize(a)
print(corpus)

## PARAGRAPH TO word
from nltk.tokenize import word_tokenize
documets= word_tokenize(a)
print(documets)

from nltk.tokenize import wordpunct_tokenize
word_tok=wordpunct_tokenize(a)
print(word_tok)

for i in corpus:
    sen_word=wordpunct_tokenize(i)
    print(sen_word)


## output for wordpunck
# ['Hello', 'Sathick', "'", 's', ',', 'How', 'are', 'you', '.']
# ['Where', 'are', 'you', 'going', 'today', '!']
# ['every', 'thing', 'is', 'alright', '.']

from nltk.tokenize import TreebankWordTokenizer

for j in corpus:
    token=TreebankWordTokenizer()    
    word=token.tokenize(j)
    print(word)

## OUTPUT FOR TREe
# ['Hello', 'Sathick', "'s", ',', 'How', 'are', 'you', '.']
# ['Where', 'are', 'you', 'going', 'today', '!']
# ['every', 'thing', 'is', 'alright', '.']