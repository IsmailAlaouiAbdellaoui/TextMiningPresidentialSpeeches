import os 
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import nltk

#os.rename("0-adams","adams")

        
def rename_folders():
    i = 0
    for (rootDir, subDirs, files) in os.walk("."):
        for subDir in subDirs:
            os.rename(subDir,str(i)+subDir)
            i += 1
     
def tokenizer_with_punctuation():
    #nltk.download('punkt')
    for (rootDir, subDirs, files) in os.walk("."):
        for file in files:
            if file == "obama_speeches_000.txt":

                with open("31obama/"+file, 'r') as content_file:
                    content = content_file.read()
                    print(word_tokenize(content))
                    return word_tokenize(content)

                        
def tokenizer_without_punctuation():
    for (rootDir, subDirs, files) in os.walk("."):
        for file in files:
            if file == "obama_speeches_000.txt":
                with open("31obama/"+file, 'r') as content_file:
                    tokenizer = RegexpTokenizer(r'\w+')
                    content = content_file.read()
                    print(tokenizer.tokenize(content))
                    return tokenizer.tokenize(content)
                
test = []
list_words = tokenizer_without_punctuation()
for word in list_words:
    test.append(word.lower())
    
print(test)
            

    
