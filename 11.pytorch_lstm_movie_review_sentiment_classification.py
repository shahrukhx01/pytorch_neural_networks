



if __name__ == "__main__":
    DATA_PATH = '/Users/shahrukh/OneDrive - Universität des Saarlandes/Saarland/WS20_21/NNTI/pytorch_deeplearning/pytorch_neural_networks/datasets/movie_sentiment_analysis'
    TRAIN_FILE_NAME = 'train.tsv'

    dataset = ReviewsDataset(DATA_PATH, TRAIN_FILE_NAME)
    dataset.load_data() ## load data to memory