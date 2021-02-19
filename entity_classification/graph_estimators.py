import torch
import torch.nn.functional as F

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from kge.graph import KGraph
from kge.training import train
from kge.negative_samplers import UniformNegativeSamplerFast
from kge.models import TransE
from kge.loss_functions import PairwiseHingeLoss



class EmbeddingEstimator(BaseEstimator):

    def __init__(self, model=None, loss=None, optimizer_cls=None, optimizer_args=None, n_negatives=2, batch_size=1000, device=None, n_epochs=500):
        self.model = model if model else TransE(100)
        self.loss = loss if loss else PairwiseHingeLoss(margin=1)
        self.n_negatives = n_negatives
        self.batch_size = batch_size
        self.optimizer_cls = optimizer_cls if optimizer_cls else torch.optim.SparseAdam
        self.optimizer_args = optimizer_args if optimizer_args else {'lr': 0.001}
        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.n_epochs = n_epochs


    def fit(self, data, y=None):
        graph, _ = data
        negative_sampler = UniformNegativeSamplerFast(self.n_negatives)

        # init components
        self.model.init(graph, self.device)
        self.loss.init(graph, self.device)
        negative_sampler.init(graph, self.device)

        self.model.train()
        
        train(
            graph,
            self.model,
            self.loss,
            negative_sampler,
            self.optimizer_cls(self.model.parameters(), **self.optimizer_args),
            self.n_epochs,
            self.device,
            self.batch_size,
            checkpoint=True,
            checkpoint_dir='checkpoints',
            checkpoint_period=10
        )

        return self


class EmbeddingTransformer(EmbeddingEstimator, TransformerMixin):

    def transform(self, data):
        _, X = data
        self.model.eval()
        with torch.no_grad():
            return self.model.entity_embedding.weight.cpu().numpy()[X]


class LinkPredictionBinaryClassifier(EmbeddingEstimator, ClassifierMixin):

    def fit(self, data, y):
        self.classes_, y = np.unique(y, return_inverse=True)

        graph, X = data

        tail = np.full_like(X, graph.n_entities)
        tail[y == 1] = graph.n_entities + 1

        graph = KGraph.from_htr(
            head=np.concatenate((graph.head, X)),
            tail=np.concatenate((graph.tail, tail)),
            relation=np.concatenate((graph.relation, np.full_like(X, graph.n_relations))),
            n_entities=graph.n_entities + 2,
            n_relations=graph.n_relations + 1
        )

        return super().fit((graph, X))

    def score(self, data):
        graph, X = data
        
        self.model.eval()
        with torch.no_grad():
            test_ids = torch.from_numpy(X).to(self.device)
            
            print(test_ids.size())

            scores = torch.empty((test_ids.size(0), 2))

            positive_scores = self.model.forward((
                test_ids,
                torch.full_like(test_ids, graph.n_entities - 1).to(self.device), 
                torch.full_like(test_ids, graph.n_relations - 1).to(self.device)
            ))

            print(positive_scores.size())

            scores[:, 0] = positive_scores

            scores[:, 1] = self.model.forward((
                test_ids,
                torch.full_like(test_ids, graph.n_entities - 2),
                torch.full_like(test_ids, graph.n_relations - 1)
            ))
        return scores


    def predict(self, data):
        D = self.score(data)
        return self.classes_[np.argmax(D, axis=1)]


    def predict_proba(self, data):
        return F.softmax(self.score(data), dim=1).numpy()

        
        

class FeatureGenerator(BaseEstimator, TransformerMixin):


    def fit(self, data, y=None):
        graph, _ = data

        self.features = np.empty((graph.n_entities, graph.n_relations * 2))

        for r in range(graph.n_relations):
            self.features[:, r] = np.bincount(graph.head[graph.relation == r], minlength=graph.n_entities)
            self.features[:, graph.n_relations + r] = np.bincount(graph.tail[graph.relation == r], minlength=graph.n_entities)

        return self

    def transform(self, data):
        _, X = data

        return self.features[X]