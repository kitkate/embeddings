import json
import pickle
import torch
import argparse
from transformers import AutoModel, AutoTokenizer


class EmbeddingsCreator:
    """Creates, saves and compares embeddings for apps"""
    MODEL = 'Salesforce/codet5p-110m-embedding'
    DEVICE = 'cpu'  # for CPU usage or "cuda" for GPU usage
    INPUT_DATASET_FILE = 'dataset.json'
    DATASET_EMBEDDINGS_FILE = 'embeddings.pickle'

    _tokenizer = None
    _model = None

    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(self.MODEL, trust_remote_code=True).to(self.DEVICE)

    def generate_embeddings_file(self, input_file):
        """Takes the json input_file containing dictionary of app names to apps and
        creates a pickle file with dictionary of app names to embeddings.
        """
        with open(input_file, 'r') as handle:
            dataset = json.load(handle)

        app_to_embedding = self.get_embeddings(dataset)

        with open(self.DATASET_EMBEDDINGS_FILE, 'wb') as handle:
            pickle.dump(app_to_embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_embeddings(self, dataset):
        """Gets dataset of app names to app descriptions and returns a dictionary of app names to embeddings.
        Since the app description is too long to be embedded from the model search_value.
        search_value contains all the vital information in a concise manner.
        """
        app_to_embedding = {}
        for name, app in dataset.items():
            json_app = json.dumps(app['search_value'])
            inputs = self._tokenizer.encode(json_app, return_tensors='pt').to(self.DEVICE)
            embedding = self._model(inputs)[0]
            app_to_embedding[name] = embedding
        return app_to_embedding

    def search_most_similar_app(self, app_file):
        """Takes json app_file with an app name to app description, generates its embeddings and
        matches it to the already saved embeddings. Prints the result.
        """
        with open(app_file, 'r') as handle:
            app = json.load(handle)
        app_to_embedding = self.get_embeddings(app)
        embedding = list(app_to_embedding.values())[0]

        with open(self.DATASET_EMBEDDINGS_FILE, 'rb') as handle:
            reference_embeddings = pickle.load(handle)
        best_similarity, most_similar_app = self.best_cosine_match(embedding, reference_embeddings)
        print(f'Best similarity with {most_similar_app} with cosine similarity {best_similarity}')

    @staticmethod
    def best_cosine_match(embedding, reference_embeddings):
        """Finds the most similar to embedding among reference_embeddings.
        Returns the cosine similarity and the name of the most similar app
        """
        cos = torch.nn.CosineSimilarity(dim=0)
        most_similar_app = ''
        best_similarity = -2
        for app, reference_embedding in reference_embeddings.items():
            similarity = cos(reference_embedding, embedding)
            if best_similarity < similarity:
                best_similarity = similarity
                most_similar_app = app
        return best_similarity, most_similar_app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finding the best match for a given App')
    parser.add_argument('-g', action='store_true',
                        help='Should the database embeddings be generated and saved to file')
    parser.add_argument('filename',
                        help='The input file containing the App to search against the database')
    args = parser.parse_args()

    embeddings_creator = EmbeddingsCreator()
    if args.g:
        embeddings_creator.generate_embeddings_file(embeddings_creator.INPUT_DATASET_FILE)
    embeddings_creator.search_most_similar_app(args.filename)

