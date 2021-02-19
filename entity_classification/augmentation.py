import time

from kge.models import TransE, DistMult, Rescal, ComplEx
from kge.loss_functions import PairwiseHingeLoss, LogisticLoss, Regularized
from kge.training import train, EpochLimit
from kge.graph import KGraph
from kge.negative_samplers import UniformNegativeSamplerFast

import torch
from torch.optim import SparseAdam

from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# from imblearn.ensemble import BalancedRandomForestClassifier

import datatable as dt
import numpy as np

# Configuration

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

train_split = 0.8

# models
models = [
#    embedding model                              loss function
    (TransE(100),                               PairwiseHingeLoss(margin=0.5)),
    (DistMult(100),                               PairwiseHingeLoss(margin=0.5)), 
    (ComplEx(100),                                PairwiseHingeLoss(margin=0.5)),
]

# optimizer
optimizer = SparseAdam
learning_rate = 0.01
batch_size = 1000
stop_condition = EpochLimit(20)

# classification
classifiers = [
    DecisionTreeClassifier(max_depth=5), 
    GaussianNB(),
    SVC(),
    SVC(class_weight='balanced')
]

n_folds = 10
metrics = [
    'accuracy',
    'average_precision',
    'f1',
    'f1_micro',
    'f1_macro'
]

# Training loop(s)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)

# results = dt.Frame(names=['sampler', 'nodes', 'edges', 'class_balance', 'model', 'loss', 'classifier'] + metrics)

# results_filename = f'classification_scores_{time.strftime("%Y%m%d_%H%M%S")}.csv'

for sampler in samplers:
    for size in sizes:
        # load graph
        graph = KGraph.from_csv(f'drive/MyDrive/social_spammer/relations_sample_{sampler}_{size}.csv', columns=[2,3,4])
        
        # load labels        
        y = dt.fread(f'drive/MyDrive/social_spammer/userdata_sample_{sampler}_{size}.csv')[:, -1].to_numpy().ravel()

        perm = np.random.permutation(graph.n_entities)
        n_train = int(graph.n_entities * train_split)
        train_ids = perm[:n_train]
        test_ids = perm[n_train:]

        tail = np.full_like(train_ids, graph.n_entities)
        tail[y[train_ids] == 1] = graph.n_entities + 1

        graph = KGraph.from_htr(
            head=np.concatenate((graph.head, train_ids)),
            tail=np.concatenate((graph.tail, tail)),
            relation=np.concatenate((graph.relation, np.full_like(train_ids, graph.n_relations))),
            n_entities=graph.n_entities + 2,
            n_relations=graph.n_relations + 1
        )

        for model, criterion in models:

            model.init(graph, device)
            criterion.init(graph, device)

            graphs = {'train': graph}
            train(
                graphs=             graphs,
                model=              model,
                criterion=          criterion,
                negative_sampler=   UniformNegativeSamplerFast(graphs, 4),
                optimizer=          optimizer(params=list(model.parameters()), lr=learning_rate),
                stop_condition=     stop_condition,
                device=             device,
                batch_size=         batch_size,
                verbose=            True,
                checkpoint=         False,
                checkpoint_period=  10
            )

            with torch.no_grad():
                model.eval()
                X = model.entity_embedding.weight.cpu().numpy()[:-2]
                
                test_ids = torch.from_numpy(test_ids).to(device)
                tail = torch.full_like(test_ids, graph.n_entities)
                tail[y[test_ids] == 1] = graph.n_entities + 1

                positive_scores = model.forward((
                    test_ids,
                    tail,
                    torch.full_like(test_ids, graph.n_relations - 1).to(device)
                ))

                print(positive_scores)

                negative_scores = model.forward((
                    test_ids,
                    torch.remainder(tail + 1, 2).to(device),
                    torch.full_like(test_ids, graph.n_relations - 1).to(device)
                ))

                correct = positive_scores > negative_scores
                print(correct)
                print(correct.mean())
                
      

            # for clf in classifiers:
            #     print(clf)
                
            #     pipeline = Pipeline([('transformer', StandardScaler()), ('estimator', clf)])
                
            #     pipeline.fit(X[train_ids], y[train_ids])

            #     print(classification_report(y[test_ids], pipeline.predict(X[test_ids])))

                # score = cross_validate(pipeline, X, y, cv=n_folds, scoring={m : m for m in metrics})
                
                # entry = dt.Frame([{
                #     'sampler': sampler,
                #     'nodes': size,
                #     'edges': len(graph),
                #     'class_balance': y.mean(),
                #     'model': str(model),
                #     'loss': str(criterion),
                #     'classifier': str(clf),
                #     **{m : np.mean(score['test_' + m]) for m in metrics}
                # }])
                # print(entry)
                # results.rbind(entry)
                # results.to_csv(results_filename)

# print(results)