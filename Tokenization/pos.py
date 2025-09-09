#####  POS (PART OF SPECH )
"""CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: “there is” … think of it like “there exists”)
FW foreign word
IN preposition/subordinating conjunction
JJ adjective – ‘big’
JJR adjective, comparative – ‘bigger’
JJS adjective, superlative – ‘biggest’
LS list marker 1)
MD modal – could, will
NN noun, singular ‘- desk’
NNS noun plural – ‘desks’
NNP proper noun, singular – ‘Harrison’
NNPS proper noun, plural – ‘Americans’
PDT predeterminer – ‘all the kids’
POS possessive ending parent’s
PRP personal pronoun –  I, he, she
PRP$ possessive pronoun – my, his, hers
RB adverb – very, silently,
RBR adverb, comparative – better
RBS adverb, superlative – best
RP particle – give up
TO – to go ‘to’ the store.
UH interjection – errrrrrrrm
VB verb, base form – take
VBD verb, past tense – took
VBG verb, gerund/present participle – taking
VBN verb, past participle – taken
VBP verb, sing. present, non-3d – take
VBZ verb, 3rd person sing. present – takes
WDT wh-determiner – which
WP wh-pronoun – who, what
WP$ possessive wh-pronoun, eg- whose
WRB wh-adverb, eg- where, when
 """

paragraph = """I have three visions for India. In 3000 years of our history, people from all over
               the world have come and invaded us, captured our lands, conquered our minds.
               From Alexander onwards, the Greeks, the Turks, the Moguls, the Portuguese, the British,
               the French, the Dutch, all of them came and looted us, took over what was ours.
               Yet we have not done this to any other nation. We have not conquered anyone.
               We have not grabbed their land, their culture,
               their history and tried to enforce our way of life on them.
               Why? Because we respect the freedom of others.That is why my
               first vision is that of freedom. I believe that India got its first vision of
               this in 1857, when we started the War of Independence. It is this freedom that
               we must protect and nurture and build on. If we are not free, no one will respect us.
               My second vision for India’s development. For fifty years we have been a developing nation.
               It is time we see ourselves as a developed nation. We are among the top 5 nations of the world
               in terms of GDP. We have a 10 percent growth rate in most areas. Our poverty levels are falling.
               Our achievements are being globally recognised today. Yet we lack the self-confidence to
               see ourselves as a developed nation, self-reliant and self-assured. Isn’t this incorrect?
               I have a third vision. India must stand up to the world. Because I believe that unless India
               stands up to the world, no one will respect us. Only strength respects strength. We must be
               strong not only as a military power but also as an economic power. Both must go hand-in-hand.
               My good fortune was to have worked with three great minds. Dr. Vikram Sarabhai of the Dept. of
               space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material.
               I was lucky to have worked with all three of them closely and consider this the great opportunity of my life.
               I see four milestones in my career"""

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import nltk
# nltk.download("averaged_perceptron_tagger_eng")
from nltk.corpus import stopwords
stop_wor=stopwords.words("english")
sent=sent_tokenize(paragraph)
for i in range(len(sent)):
    words=word_tokenize(sent[i])
    words=[word for word in words if word not in set(stop_wor) ]
    pos_tag=nltk.pos_tag(words)
    print(pos_tag)


a="Taj is beautiful"
print(nltk.pos_tag(a.split()))

### pos_tag() → takes one list of words.

##pos_tag_sents() → takes a list of lists of words (multiple sentences).

import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

text = "The quick brown fox jumps over the lazy dog. I love Python NLP."

# Step 1: Split text into sentences
sentences = nltk.sent_tokenize(text)

# Step 2: Tokenize each sentence into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Step 3: Remove stopwords
stop_wor = set(stopwords.words("english"))
filtered_sentences = [[word for word in sent if word.lower() not in stop_wor] for sent in tokenized_sentences]

# Step 4: POS tagging for all sentences at once
tagged_sentences = nltk.pos_tag_sents(filtered_sentences)

print(tagged_sentences)
