
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchtext.data import Field, TabularDataset, BucketIterator, LabelField
import spacy
import pandas as pd
import tqdm


"""
Here we use torchtext based dataloader to perform following steps:
1. preprocess data
2. tokenize data
3. create vocabulary
4. create batches

References:
https://www.youtube.com/watch?v=KRgq4VnCr7I&ab_channel=AladdinPersson
http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/

"""




class ReviewsDataset():
    def __init__(self, data_path, train_path):

        ## write the tokenizer
        tokenize = lambda review : review.split()
        ## define your fields for ID filed you can use RAWField class
        self.TEXT  = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
        self.LABEL  = LabelField()
        

        
        self.fields = [("PhraseId", None), # we won't be needing the id, so we pass in None as the field
                 ("SentenceId", None), ("Phrase", self.TEXT),
                 ("Sentiment", self.LABEL)] #{ 'Phrase': ('r', self.review), 'Sentiment': ('s', self.sentiment) }
        ## set paths
        self.data_path = data_path
        self.train_path = train_path

    def load_data(self):
        self.train_data = TabularDataset.splits(
            path='{}'.format(self.data_path),
            train='{}'.format(self.train_path),
            format='tsv',
            fields=self.fields)[0]
        
        self.TEXT.build_vocab(self.train_data, max_size=10000, min_freq=1)
        self.LABEL.build_vocab(self.train_data)
        self.train_iterator, _ = BucketIterator.splits((self.train_data, None), 
                                                    batch_sizes=(64, 64), 
                                                    sort_within_batch=True,
                                                    sort_key=lambda x: len(x.Phrase))


    def __str__(self):
        return 'review: {} \n sentiment: {}'.format(self.train_data[0][0].__dict__['r'], self.train_data[0][0].__dict__['s'])


           
        
 ## doing LSTM based classification       
class SimpleLSTMBaseline(nn.Module):
    def __init__(self, hidden_dim, emb_dim=300, num_linear=1, ds=None):
        super().__init__() # don't forget to call this!
        self.embedding = nn.Embedding(len(ds.TEXT.vocab), emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1)
        self.linear_layers = []
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers = nn.ModuleList(self.linear_layers)
        self.predictor = nn.Linear(hidden_dim, 5)

    def forward(self, seq):
        preds = None
        hdn, _ = self.encoder(self.embedding(seq))
        feature = hdn[-1, :, :]
        for layer in self.linear_layers:
          feature = layer(feature)
          preds = self.predictor(feature)
        return preds



class BatchWrapper:
      def __init__(self, dl, x_var, y_var):
            self.dl, self.x_var, self.y_var = dl, x_var, y_var # we pass in the list of attributes for x 

      def __iter__(self):
            for batch in self.dl:
                  x = getattr(batch, self.x_var) # we assume only one input in this wrapper and one output
                  y =  getattr(batch, self.y_var)
                  yield (x, y)

      def __len__(self):
            return len(self.dl)



if __name__ == "__main__":
    DATA_PATH = '/Users/shahrukh/OneDrive - Universität des Saarlandes/Saarland/WS20_21/NNTI/pytorch_deeplearning/pytorch_neural_networks/datasets/movie_sentiment_analysis'
    TRAIN_FILE_NAME = 'train.tsv'

    dataset = ReviewsDataset(DATA_PATH, TRAIN_FILE_NAME)
    dataset.load_data() ## load data to memory
    em_sz = 100
    nh = 500
    nl = 3

    train_dl = BatchWrapper(dataset.train_iterator, "Phrase", "Sentiment")
    model = SimpleLSTMBaseline(nh, emb_dim=em_sz, ds=dataset, num_linear=nl)


    opt = optim.Adam(model.parameters(), lr=1e-2)
    loss_func = nn.CrossEntropyLoss()

    epochs = 2

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_corrects = 0
        model.train() # turn on training mode
        for x, y in tqdm.tqdm(train_dl): # thanks to our wrapper, we can intuitively iterate over our data!
            if(sum(y>4).item()):
                y[(y>4).nonzero().data[0][0].item()] = 4
            opt.zero_grad()

            preds = model(x)
            loss = loss_func(preds, y )
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(dataset.train_data)

        print('Epoch: {}, Training Loss: {:.4f}'.format(epoch, epoch_loss))