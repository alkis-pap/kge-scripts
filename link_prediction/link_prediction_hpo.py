from kge.graph import KGraph
from kge.models import TransE, DistMult, Rescal, ComplEx
from kge.loss_functions import LogisticLoss, PairwiseHingeLoss, Regularized, NLLMulticlass
from kge.negative_samplers import UniformNegativeSamplerFast
from kge.utils import timeit, strip_whitespace
from kge.sklearn import EmbeddingEstimator

import torch
from torch.optim import SGD, SparseAdam, Adam

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, PredefinedSplit

import pandas as pd

import uuid, copy, os, functools, operator


datasets = {
    'FB15K-237': [
        '~/Data/FB15k-237/train.txt',
        '~/Data/FB15k-237/valid.txt',
        '~/Data/FB15k-237/test.txt'
    ]
    # ,
    # 'WN18': [
    #     '~/Data/WN18/train.txt',
    #     '~/Data/WN18/valid.txt',
    #     '~/Data/WN18/test.txt'
    # ],
    # 'WN18RR': [
    #     '~/Data/WN18RR/train.txt',
    #     '~/Data/WN18RR/valid.txt',
    #     '~/Data/WN18RR/test.txt'
    # ],
    # 'FB15K': [
    #     '~/Data/FB15k/train.txt',
    #     '~/Data/FB15k/valid.txt',
    #     '~/Data/FB15k/test.txt'
    # ]
}

estimator = EmbeddingEstimator(
    TransE(),
    optimizer_cls=Adam,
    batch_size=5000,
    verbose=True
)

common_values = {
    'optimizer_args': [dict(lr=lr) for lr in [0.001, 0.0005, 0.0001, 0.00005]],
    'n_negatives': [10, 30, 50],
    'model__embedding_dim': [50, 100, 200]
}

param_values = [
    {
        **common_values,
        'loss': [
            PairwiseHingeLoss(1), 
            PairwiseHingeLoss(.5), 
            NLLMulticlass(), 
            LogisticLoss()
        ]
    },
    {
        **common_values,
        'loss': [
            Regularized(PairwiseHingeLoss(1)),
            Regularized(PairwiseHingeLoss(.5)), 
            Regularized(NLLMulticlass()), 
            Regularized(LogisticLoss())],
        'model__normalize_embeddings': [False],
        'loss__l_ent': [0.1, 0.01, 0.001, 0.0001]
    }
]

def score(estimator, data):
    return estimator.evaluate(data[0])['both']['hits@10']


for dataset, files in datasets.items():
    print(dataset)

    g_train, g_valid, g_test = KGraph.from_csv(files, columns=[0,2,1])
    g_train.add_inverse_triples()

    X = [g_train, g_valid]

    search = HalvingRandomSearchCV(
        estimator, 
        param_values, 
        factor=2, 
        resource='n_epochs',
        min_resources=50,
        max_resources=800, 
        cv=PredefinedSplit([-1, 0]),
        scoring=score,
        random_state=0,
        refit=False,
        verbose=100,
        return_train_score=False
    )

    search.fit(X)

    print(search.best_score_)
    print(search.best_params_)