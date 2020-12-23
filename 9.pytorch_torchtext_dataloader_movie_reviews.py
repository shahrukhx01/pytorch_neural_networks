from torchtext.data import Field, TabularDataset, BucketIterator, LabelField
import spacy
import pandas as pd

"""
Here we use torchtext based dataloader to perform following steps:
1. preprocess data
2. tokenize data
3. create vocabulary
4. create batches
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


if __name__ == "__main__":
    DATA_PATH = '/Users/shahrukh/OneDrive - Universität des Saarlandes/Saarland/WS20_21/NNTI/pytorch_deeplearning/pytorch_neural_networks/datasets/movie_sentiment_analysis'
    TRAIN_FILE_NAME = 'train.tsv'

    dataset = ReviewsDataset(DATA_PATH, TRAIN_FILE_NAME)
    dataset.load_data() ## load data to memory
    
       
        
        