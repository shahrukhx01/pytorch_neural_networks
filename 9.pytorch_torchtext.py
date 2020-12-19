from torchtext.data import Field, TabularDataset, BucketIterator, LabelField


class ReviewsDataset():
    def __init__(self, data_path, train_path):

        ## write the tokenizer
        tokenize = lambda review : review.split()
        ## define your fields for ID filed you can use RAWField class
        review  = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
        sentiment  = LabelField()
        

        
        self.fields = { 'Phrase': ('r', review), 'Sentiment': ('s', sentiment) }
        ## set paths
        self.data_path = data_path
        self.train_path = train_path

    def load_data(self):
        self.train_data = TabularDataset.splits(
            path='{}'.format(self.data_path),
            train='{}'.format(self.train_path),
            format='tsv',
            fields=self.fields)

    def __str__(self):
        return 'review: {} \n sentiment: {}'.format(self.train_data[0][0].__dict__['r'], self.train_data[0][0].__dict__['s'])


if __name__ == "__main__":
    DATA_PATH = '/Users/shahrukh/OneDrive - Universität des Saarlandes/Saarland/WS20_21/NNTI/pytorch_deeplearning/pytorch_neural_networks/datasets/movie_sentiment_analysis'
    TRAIN_FILE_NAME = 'train.tsv'

    dataset = ReviewsDataset(DATA_PATH, TRAIN_FILE_NAME)
    dataset.load_data() ## load data to memory
    print(str(dataset))