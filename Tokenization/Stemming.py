word=["running", "runner", "runs", "easily", "fairly", "caresses","writting","files","making","connected"]
b=[]

from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
for i in word:
    b.append(stemmer.stem(i))
# print(b)

from nltk.stem import RegexpStemmer
re_stem=RegexpStemmer("ing$|e$|s$|able$",min=4)    ### like removing the last character like ing or e or s or able regex pattern is focus only the last value
print(re_stem.stem("ingeating"))

re_stem=RegexpStemmer("ing|e$|s$|able$",min=4)   ### like removing the last character like ing or e or s or able regex pattern is focus only the ALL value
print(re_stem.stem("ingeating"))


### REGULAR EXPRASSION IS USIGN THE $ SYMBOL IS ONLY REMOVED ONE VALUES BUT IS NOT AVAILABLE THE ENTIRE WORD WILL BE REMOVED

### snowball stemming

from nltk.stem import SnowballStemmer

words = ["running", "runner", "flies", "easily", "fairly", "caresses", "organization"]

c=[]  #['run', 'runner', 'fli', 'easili', 'fair', 'caress', 'organ']
snow_Stem=SnowballStemmer("english")     ### langane is import one for 
for i in words:
    c.append(snow_Stem.stem(i))    # the output is different between like porter stemmer is like fairly to fairli but snow ball is fair is best one
print(c)                            # like simar to orgnization to organ 