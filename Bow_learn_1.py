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
from nltk.stem import PorterStemmer
import re
ps=PorterStemmer()

corpus=[]
for i in range(0,len(message)):
    review=re.sub('[^a-zA-z]',' ',message['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in stopwords.words("english")]
    review=" ".join(review)
    corpus.append(review)
for y in corpus:
    print(y)
# print(corpus)


## Create BAG OF WORDS

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=100,binary=True)

x=cv.fit_transform(corpus).toarray()
print(x)


