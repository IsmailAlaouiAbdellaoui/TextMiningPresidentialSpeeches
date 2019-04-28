import os 
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

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

fill_all_speeches()
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




test = stemmer()
print(len(test))

    

            

    
