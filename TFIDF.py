import pandas as pd

message = pd.read_csv(
    r"C:\Users\Admin\Desktop\Learning\ML\data_set\spam.csv",
    encoding="latin-1"
)

# Select first two columns (v1, v2)
message = message[['v1', 'v2']]

# Rename columns
message.columns = ['label', 'message']

# print(message.head())

## Data clean processing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
ps=WordNetLemmatizer()

corpus=[]
for i in range(0,len(message)):
    review=re.sub('[^a-zA-z]',' ',message['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.lemmatize(word) for word in review if word not in stopwords.words("english")]
    review=" ".join(review)
    corpus.append(review)
# for y in corpus:
#     print(y)
# print(corpus)


## Create BAG OF WORDS

from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer(max_features=100)

x=cv.fit_transform(corpus).toarray()
# print(x)


## N Grams

from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer(max_features=100,ngram_range=(2,2))

x=cv.fit_transform(corpus).toarray()
# print(x)
print(cv.vocabulary_)


