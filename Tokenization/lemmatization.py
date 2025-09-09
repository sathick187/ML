## its maily used for Q&A, CHATSBOTS,...

from nltk.stem import WordNetLemmatizer
# import nltk
# nltk.download('wordnet')
lem=WordNetLemmatizer()
words = ["running", "runner", "flies", "easily", "fairly", "caresses", "organization","better","going"]
c=[]  

for i in words:
    c.append(lem.lemmatize(i))   # ouput ['running', 'runner', 'fly', 'easily', 'fairly', 'caress', 'organization', 'better', 'going']]
print(c)

# the lemmatizie is defalut of pos in noun like pos="n"

# for i in words:
#     c.append(lem.lemmatize(i,pos="v"))  # op ['run', 'runner', 'fly', 'easily', 'fairly', 'caress', 'organization', 'better', 'go']
# print(c)

# the lemmatizie is second one verb the output is different between noun