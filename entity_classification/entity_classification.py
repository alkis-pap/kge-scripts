import uuid

from graph_estimators import *

from kge.models import TransE, DistMult, Rescal, ComplEx
from kge.loss_functions import PairwiseHingeLoss, LogisticLoss, Regularized
from kge.graph import KGraph
from kge.utils import timeit

import torch
from torch.optim import SparseAdam

from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

# from imblearn.ensemble import BalancedRandomForestClassifier

import datatable as dt
import pandas as pd
import numpy as np

# Configuration

base_dir = '~/Data/social_spammer'

# graph sampling method
samplers = [
    'RandomWalkSampler',
    'ForestFireSampler',
    'FrontierSampler'
]

sizes = [
    10000,
    50000,
    100000
]

train_split = 0.9

# estimators
estimators = []

classifiers = [
    GaussianNB(), 
    DecisionTreeClassifier(),
    SVC(),
    KNeighborsClassifier()
]

for clf in classifiers:
    # handcrafted features estimators
    estimators.append(Pipeline([
        ('embedding', FeatureGenerator()),
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ]))

# knowledge graph embedding hyper-parameters
n_negatives = 10
optimizer = SparseAdam
learning_rate = 0.001
batch_size = 1000
n_epochs = 500

for model_cls in [TransE, DistMult, Rescal, ComplEx]:
    for d in [50, 100, 200]:
        for clf in classifiers:
            # unsupervised embedding estimators
            estimators.append(Pipeline([
                ('embedding', EmbeddingTransformer(
                    model=model_cls(d), 
                    optimizer_cls=optimizer, 
                    optimizer_args= dict(lr=learning_rate), 
                    batch_size=batch_size, 
                    n_epochs=n_epochs
                )),
                ('scaler', StandardScaler()),
                ('classifier', clf)
            ]))

        # link prediction estimators
        estimators.append(LinkPredictionBinaryClassifier(
            model=model_cls(d), 
            optimizer_cls=optimizer, 
            optimizer_args= dict(lr=learning_rate), 
            batch_size=batch_size, 
            n_epochs=n_epochs
        ))

metrics = [
    'accuracy',
    'average_precision',
    'f1',
    'f1_micro',
    'f1_macro'
]

results = []
results_filename = 'results/results_' + uuid.uuid4().hex + '.csv'

for sampler in samplers:
    for size in sizes:
        # load graph
        graph = KGraph.from_csv(f'{base_dir}/relations_sample_{sampler}_{size}.csv', columns=[2,3,4])
        
        # load labels        
        y = dt.fread(f'{base_dir}/userdata_sample_{sampler}_{size}.csv')[:, -1].to_numpy().ravel()

        perm = np.random.permutation(graph.n_entities)
        n_train = int(graph.n_entities * train_split)
        train_ids = perm[:n_train]
        test_ids = perm[n_train:]

        for estimator in estimators:
            print(estimator)
            estimator.fit((graph, train_ids), y[train_ids])
            
            y_test = y[test_ids]

            y_pred = estimator.predict((graph, test_ids))

            try:
                scores = estimator.predict_proba((graph, test_ids))[:, 0].ravel()
            except AttributeError as e:
                print(e)
                scores = estimator.decision_function((graph, test_ids))
            
            results.append({
                'Sampler' : sampler,
                'Size' : size,
                'Estimator': ' '.join(str(estimator).split()),
                'Acc' : accuracy_score(y_test, y_pred),
                'AP' : average_precision_score(y_test, scores),
                'AUROC' : roc_auc_score(y_test, scores)
            })

            # print and save results
            df = pd.DataFrame(results)
            print(df.iloc[[-1]])
            df.to_csv(results_filename, index=False)

