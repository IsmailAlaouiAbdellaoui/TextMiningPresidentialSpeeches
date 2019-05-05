import os 
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import gensim
import time

#os.rename("0-adams","adams")
adams = [[]]
arthur = [[]]
bharisson = [[]]
buchanan = [[]]
bush = [[]]
carter = [[]]
cleveland = [[]]
clinton = [[]]
coolidge = [[]]
eisenhower = [[]]
fdroosevelt = [[]]
fillmore = [[]]
ford = [[]]
garfield = [[]]
grant = [[]]
gwbush = [[]]
harding = [[]]
harrison = [[]]
hayes = [[]]
hoover = [[]]
jackson = [[]]
jefferson = [[]]
johnson = [[]]
jqadams = [[]]
kennedy = [[]]
lbjohnson = [[]]
lincoln = [[]]
madison = [[]]
mckinley = [[]]
monore = [[]]
nixon = [[]]
obama = [[]]
pierce = [[]]
polk = [[]]
reagan = [[]]
roosevelt = [[]]
taft = [[]]
taylor = [[]]
truman = [[]]
tyler = [[]]
vanburen = [[]]
washington = [[]]
wilson = [[]]

all_speeches = [] 

folder_names = []
tokenized_all_speeches_punctuation = []
tokenized_all_speeches_without_punctuation = []
speeches_without_stopwords = []
speeches_after_stemming = []

def fill_all_speeches():
    for (rootDir, subDirs, files) in os.walk("."):
        folder_names.append(rootDir)
    for i in range(1,len(folder_names)):
        for (rootDir, subDirs, files) in os.walk(folder_names[i]):
                for file in files:
                    if file.endswith(".txt"):
                        with open(str(folder_names[i])+"\\"+str(file), 'r', encoding="utf8") as content_file:
                            content = content_file.read()
                            all_speeches.append(content)

#fill_all_speeches()
#print(len(all_speeches))


        
def rename_folders():
    i = 0
    for (rootDir, subDirs, files) in os.walk("."):
        for subDir in subDirs:
            os.rename(subDir,str(i)+subDir)
            i += 1
     
def tokenizer_with_punctuation():
    nltk.download('punkt')
    for speech in all_speeches:
        tokenized_all_speeches_punctuation.append(word_tokenize(speech))
    
#tokenizer_with_punctuation()


                        
def tokenizer_without_punctuation():
    tokenizer = RegexpTokenizer(r'\w+')
    for speech in all_speeches:
        tokenized_all_speeches_without_punctuation.append(tokenizer.tokenize(speech))
    
    return tokenized_all_speeches_without_punctuation
    
#test = tokenizer_without_punctuation()
#print(test[0])
                
def delete_stopwords():
#    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    temp = []
    list_words = tokenizer_without_punctuation()
    for i in range(len(list_words)):
        for j in range(len(list_words[i])):
            list_words[i][j] = list_words[i][j].lower()

    for i in range(len(list_words)):
            temp.append([word for word in list_words[i] if word not in stop_words])
            

    return temp
            

#delete_stopwords()

def stemmer():
    words_without_stopwords = delete_stopwords()
    porter = PorterStemmer()
    for i in range(len(words_without_stopwords)):
        for j in range(len(words_without_stopwords[i])):
            words_without_stopwords[i][j] = porter.stem(words_without_stopwords[i][j])
            
    return words_without_stopwords



#t0 = time.process_time()
#words = stemmer()
def word2vecspeeches(stemmed_words_speeches):
    model = gensim.models.Word2Vec(stemmed_words_speeches,size=150,window=10,min_count=2,workers=10)
    model.train(stemmed_words_speeches, total_examples=len(stemmed_words_speeches), epochs=10)
#    w1 = ["sky"]
#    print(model.wv.most_similar(positive=w1))

#with open("opinion-lexicon-English/negative-words.txt", "r", encoding="utf8") as negative_file:
#    content = negative_file.read()
##    all_speeches.append(content) 
#    print(content)
negative_list = []
negative_file = open("opinion-lexicon-English/negative-words.txt", "r")
content = negative_file.readlines()
for i in range(len(content)):
    if i > 32:
        negative_list.append(content[i])
negative_file.close()
print(len(negative_list))


positive_list = []
positive_file = open("opinion-lexicon-English/positive-words.txt", "r")
content = positive_file.readlines()
for i in range(len(content)):
    if i > 30:
        positive_list.append(content[i])
positive_file.close()
print(len(positive_list))



#def word2vecnegatives(stemmed_words_negatives):
    
#To do list:
#-Combine negative/positive into 1 list
#-Transform each word into a vector using Word2Vec
#-Create a big list of scores ( for each word)
#-Training SVM (or LSTM) using this newly created dataset
#-Use trained model to test on our speeches
    

            

    
