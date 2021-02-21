import os, subprocess

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
import numpy as np



if not os.path.isfile('link-prediction.zip'):
    subprocess.run(['wget', 'https://www.dropbox.com/s/6066tiybyro1jmh/link-prediction.zip'])
subprocess.run(['unzip', '-u', 'link-prediction.zip', '-d', 'data'])
subprocess.run(['mkdir', '-p', 'checkpoints'])

datasets = {
    'FB15K-237': [
        'data/FB15k-237/train.txt',
        'data/FB15k-237/valid.txt',
        'data/FB15k-237/test.txt'
    ]
    # ,
    # 'WN18': [
    #     'data/WN18/train.txt',
    #     'data/WN18/valid.txt',
    #     'data/WN18/test.txt'
    # ],
    # 'WN18RR': [
    #     'data/WN18RR/train.txt',
    #     'data/WN18RR/valid.txt',
    #     'data/WN18RR/test.txt'
    # ],
    # 'FB15K': [
    #     'data/FB15k/train.txt',
    #     'data/FB15k/valid.txt',
    #     'data/FB15k/test.txt'
    # ]
}

estimator = EmbeddingEstimator(
    TransE(),
    optimizer_cls=Adam,
    batch_size=5000,
    verbose=True,
    checkpoint_dir='checkpoints',
    validation=True,
    validation_period=25,
    patience=4,
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
    train_graph, test_graph = data[0]
    return estimator.evaluate(test_graph, train_graph)['both']['hits@10']


for dataset, files in datasets.items():
    print(dataset)

    g_train, g_valid, g_test = KGraph.from_csv(files, columns=[0,2,1])
    
    np.random.seed(0)
    g_valid_valid, g_valid_test = g_valid.random_split(0.1, 0.9)
    
    g_augmented = g_train.with_inverse_triples()

    X = [(g_augmented, g_valid_valid), (g_train, g_valid_test)]

    g_train_final = g_train.combine(g_valid)
    g_augmented_final = g_train_final.with_inverse_triples()


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

    print('Best validation score:', search.best_score_)
    print('Best parameters:', search.best_params_)

    search.best_estimator_.fit([g_augmented_final])
    
    print('Test score:', search.best_estimator_.evaluate(g_test, g_train_final))