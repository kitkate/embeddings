import math
import unittest
import torch
from embeddings import EmbeddingsCreator
from unittest.mock import patch, Mock


class TestEmbeddings(unittest.TestCase):

    def test_get_embeddings(self):
        dataset = {'App1': {'search_value': 'Contact (Address)'}}

        def __init__(self):
            self._tokenizer = Mock()
            self._tokenizer.encode = Mock()
            self._tokenizer.encode.return_value = Mock()
            self._tokenizer.encode.return_value.to = Mock()
            self._tokenizer.encode.return_value.to.return_value = 0
            self._model = Mock(return_value=[torch.tensor([1., 2., 3.])])

        with patch.object(EmbeddingsCreator, '__init__', __init__):
            embeddings_creator = EmbeddingsCreator()
            app_to_embedding = embeddings_creator.get_embeddings(dataset)
            self.assertEqual('App1', list(app_to_embedding.keys())[0], 'App1 should have embeddings')

    def test_best_cosine_match(self):
        reference_embeddings = {'App1': torch.tensor([2., 4., 6.]), 'App2': torch.tensor([1., 1., 1.])}
        embedding = torch.tensor([1., 2., 3.])
        best_similarity, most_similar_app = EmbeddingsCreator.best_cosine_match(embedding, reference_embeddings)
        expected_similarity = 1.
        actual_similarity = best_similarity.item()
        self.assertTrue(math.isclose(expected_similarity, actual_similarity, abs_tol=0.0001),
                        f'The cosine similarity is {actual_similarity} and not {expected_similarity}')
        self.assertEqual('App1', most_similar_app, 'The most similar app is not as expected')


if __name__ == '__main__':
    unittest.main()
