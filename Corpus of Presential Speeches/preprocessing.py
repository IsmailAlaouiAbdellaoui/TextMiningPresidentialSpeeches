import os 
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import gensim
#import time
import numpy as np
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
#negative_list = []
#negative_file = open("opinion-lexicon-English/negative-words.txt", "r")
#content = negative_file.readlines()
#for i in range(len(content)):
#    if i > 32:
#        negative_list.append(content[i])
#negative_file.close()
#print(len(negative_list))
#
#
#positive_list = []
#positive_file = open("opinion-lexicon-English/positive-words.txt", "r")
#content = positive_file.readlines()
#for i in range(len(content)):
#    if i > 30:
#        positive_list.append(content[i])
#positive_file.close()
#print(len(positive_list))
from string import punctuation
from collections import Counter

amazon_reviews = []
amazon_labels = []

amazon_file = open("amazon_cells_labelled.txt","r")
content = amazon_file.readlines()
amazon_file.close()
for x in content:
    temp = x.split('\t')
    amazon_reviews.append(''.join([c for c in temp[0].lower() if c not in punctuation]))
    amazon_labels.append(temp[1].replace('\n',''))
    

all_text2 = ' '.join(amazon_reviews)
# create a list of words
words = all_text2.split()
# Count all the words using Counter Method
count_words = Counter(words)

total_words = len(words)
sorted_words = count_words.most_common(total_words)
#print(sorted_words)
vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}
#print(vocab_to_int)
reviews_int = []
for review in amazon_reviews:
    r = [vocab_to_int[w] for w in review.split()]
    reviews_int.append(r)
    
encoded_labels = [1 if label =='1' else 0 for label in amazon_labels]
encoded_labels = np.array(encoded_labels)
#print(encoded_labels)


def pad_features(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_length), dtype = int)
    
    for i, review in enumerate(reviews_int):
        review_len = len(review)
        
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = zeroes+review
        elif review_len > seq_length:
            new = review[0:seq_length]
        
        features[i,:] = np.array(new)
    
    return features

len_feat = 10
features = pad_features(reviews_int,len_feat)

split_frac = 0.8
train_x = features[0:int(split_frac*10)]
train_y = encoded_labels[0:int(split_frac*len_feat)]
remaining_x = features[int(split_frac*len_feat):]
remaining_y = encoded_labels[int(split_frac*len_feat):]
valid_x = remaining_x[0:int(len(remaining_x)*0.5)]
valid_y = remaining_y[0:int(len(remaining_y)*0.5)]
test_x = remaining_x[int(len(remaining_x)*0.5):]
test_y = remaining_y[int(len(remaining_y)*0.5):]

import torch
from torch.utils.data import DataLoader, TensorDataset
# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
# dataloaders
batch_size = 50
# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()
print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size()) # batch_size
print('Sample label: \n', sample_y)

import torch.nn as nn

class SentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                            dropout=drop_prob, batch_first=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
    
        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
#        if (train_on_gpu):
#            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
#                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
#        else:
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden


# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int)+1 # +1 for the 0 padding
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2
net = SentimentLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
print(net)

# loss and optimization functions
lr=0.001

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


# training params

epochs = 4 # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip=5 # gradient clipping

# move model to GPU, if available
#if(train_on_gpu):
#    net.cuda()

net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

#        if(train_on_gpu):
#            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        inputs = inputs.type(torch.LongTensor)
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

#                if(train_on_gpu):
#                    inputs, labels = inputs.cuda(), labels.cuda()

                inputs = inputs.type(torch.LongTensor)
                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

#def word2vecnegatives(stemmed_words_negatives):
    
#To do list:
#-Combine negative/positive into 1 list
#-Transform each word into a vector using Word2Vec
#-Create a big list of scores ( for each word)
#-Training SVM (or LSTM) using this newly created dataset
#-Use trained model to test on our speeches
    

            

    
