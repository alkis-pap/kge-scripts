import os, time, math, uuid
from itertools import groupby
from operator import itemgetter

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import pandas as pd
import datatable as dt

from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL

import dgl
from rgcn import EntityClassify, RelGraphEmbed

import matplotlib.pyplot as plt

from sklearn import metrics

base_dir = 'social_spammer'

# graph sampling method
samplers = [
    'RandomWalkSampler',
    'ForestFireSampler'
    ,
    'FrontierSampler'
]

sizes = [
    10000
    ,
    50000
]

train_ratio = 0.9

MIXED_PRECISION = False

DEVICE = 'cuda:0'


def read_data(relations_file, userdata_file):
    relations = dt.fread(relations_file, columns=[False,False,True,True,True]).to_numpy()

    rel_ids = np.unique(relations[:, 2])

    data = {}

    for rel_id in rel_ids:
        edges = relations[relations[:, 2] == rel_id, 0:2]
        data[('user', f'rel_{rel_id}', 'user')] = (edges[:, 0], edges[:, 1])

    g = dgl.heterograph(data)
    
    userdata = dt.fread(userdata_file, columns=[False,True,True,True,True])

    y = torch.tensor(userdata[:, -1].to_numpy().ravel(), dtype=torch.long).detach()

    return g, y


roc_displays = []
prc_displays = []

def classification_scores(y_true, y_pred, scores, name):
    
    # ROC Curve
    fpr, tpr, thresholds = metrics.roc_curve(y_true, scores)
    roc_auc = metrics.auc(fpr, tpr)
    roc_displays.append(metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name))
    # plt.savefig(roc_path)

    # PR Curve
    precision, recall, _ = metrics.precision_recall_curve(y_true, scores)
    ap = metrics.average_precision_score(y_true, scores)
    prc_displays.append(
      metrics.PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=ap, estimator_name=name)
    )
    # plt.savefig(prc_path)

    return {
        'accuracy' : metrics.accuracy_score(y_true, y_pred),
        'balanced_accuracy' : metrics.balanced_accuracy_score(y_true, y_pred),
        'average_precision' : ap,
        'roc_auc' : roc_auc
        # ,
        # 'roc_curve' : roc_path,
        # 'pr_curve' : prc_path
    }


class RGCNWorker():

    def __init__(self, graph, labels, train_ids, valid_ids, test_ids):
        self.graph = graph
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.labels = labels
        self.train_ids = train_ids
        self.valid_ids = valid_ids
        self.test_ids = test_ids
        

    def compute(self, config, test=False, name="RGCN"):
        
        # print(config)

        if test:
            train_ids = np.concatenate((self.train_ids, self.valid_ids))
            n_train = int(len(train_ids) * (1 - config.get('early_stopping_split', 0.1)))
            valid_ids = torch.tensor(train_ids[n_train:]).to(DEVICE)
            train_ids = torch.tensor(self.train_ids[:n_train]).to(DEVICE)
            test_ids = torch.tensor(self.valid_ids).to(DEVICE)
        else:
            n_train = int(len(self.train_ids) * (1 - config.get('early_stopping_split', 0.1)))
            train_ids = torch.tensor(self.train_ids[n_train:]).to(DEVICE)
            valid_ids = torch.tensor(self.train_ids[:n_train]).to(DEVICE)
            test_ids = torch.tensor(self.valid_ids).to(DEVICE)

        labels = self.labels.to(DEVICE)


        model = EntityClassify(
            self.graph.to(DEVICE),
            int(config['hidden_dim']),
            out_dim=2,
            num_bases=int(config['num_bases']),
            num_hidden_layers=config.get('num_hidden_layers', 0),
            dropout=config['dropout'],
            use_self_loop=config['use_self_loop']
        ).to(torch.device(DEVICE), torch.float32)

        optimizer = Adam(model.parameters(), lr=config['learning_rate'])

        scaler = GradScaler()

        best_score = None

        for epoch in range(1000):
            model.train()
            optimizer.zero_grad()
            with autocast(enabled=MIXED_PRECISION):
                logits = model()['user']
                loss = F.cross_entropy(logits[train_ids], labels[train_ids])
            
            if MIXED_PRECISION:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            

            model.eval()
            with torch.no_grad():
                with autocast(enabled=MIXED_PRECISION):
                    logits = model()['user'][valid_ids]

                try:
                    
                    val_score = metrics.average_precision_score(self.labels[valid_ids], logits[:,1].cpu())
                except ValueError:
                    if best_score:
                        break
                    print('fail')
                    print(torch.max(logits))
                    return { 'loss': 0, 'status': STATUS_FAIL }

                if not best_score or val_score > best_score:

                    # if best_score and epoch - best_epoch > 2:
                    #     print(f'COMEBACK: {epoch - best_epoch}')

                    best_score = val_score
                    best_logits = logits
                    best_epoch = epoch
                elif epoch > best_epoch + config.get('patience', 20):
                    print('out of patience')
                    break
                    
            # print(loss.item())

        print(f'epochs: {best_epoch}')

        with torch.no_grad():
            with autocast():
                logits = best_logits[test_ids].cpu()

            y_pred = logits.argmax(dim=1)
            y_true = self.labels[test_ids]
            scores = logits[:,1]

            # train_acc = torch.sum(logits[train_ids].argmax(dim=1) == self.labels[train_ids]).item() / len(train_ids)
            # test_loss = F.cross_entropy(logits[test_ids], self.labels[test_ids])

            if not test:
                # test_acc = torch.sum(y_pred == y_true).item() / len(test_ids)

                try:
                    test_score = metrics.average_precision_score(y_true, scores)

                except ValueError:
                    print(torch.max(logits))
                    raise

                return { 'loss': -test_score, 'status': STATUS_OK }
            else:
                return classification_scores(self.labels[test_ids], y_pred, scores, name)
                




results_file = 'rgcn_hyperopt.csv'
results = []
# if os.path.isfile(results_file):
#     results = pd.read_csv(results_file).to_dict('records')

for sampler in samplers:
    for size in sizes:
        
        g, y = read_data(f'{base_dir}/relations_sample_{sampler}_{size}.csv', f'{base_dir}/userdata_sample_{sampler}_{size}.csv')

        train_split_file = f'{base_dir}/train_split_{sampler}_{size}.npz'

        try:
            train_split = np.load(train_split_file)
            train_ids = train_split['train_ids']
            valid_ids = train_split['valid_ids']
            test_ids = train_split['test_ids']
        except:
            perm = np.random.permutation(g.num_nodes())
            n_train = int(len(perm) * .8)
            n_test = int(len(perm) * .1)
            train_ids = perm[ : n_train]
            valid_ids = perm[n_train : n_train + n_test]
            test_ids = perm[n_train + n_test : ]
            np.savez_compressed(train_split_file, train_ids=train_ids, valid_ids=valid_ids, test_ids=test_ids)
            
        
        entry = {
            'sampler': sampler,
            'size': size
        }

        # if results and any(all(r.get(key) == val for key, val in entry.items()) for r in results):
        #     print('skipping')
        #     continue

        w = RGCNWorker(g, y, train_ids, valid_ids, test_ids)

        fspace = {
            'hidden_dim': hp.quniform('hidden_dim', 50, 300, 10),
            'num_bases': hp.quniform('num_bases', 1, len(g.etypes), 1),
            # 'num_hidden_layers': hp.choice('num_hidden_layers', [0, 1]),
            'dropout': hp.uniform('dropout', 0, .8),
            'use_self_loop': hp.choice('use_self_loop', [True, False]),
            'learning_rate': hp.loguniform('learning_rate', math.log(1e-4), math.log(2))
            ,
            'early_stopping_split': hp.loguniform('early_stopping_split', math.log(0.01), math.log(.15))
            # ,
            # 'n_epochs' : hp.uniform('n_epochs', 15, 80)
        }

        best_config = fmin(w.compute, fspace, algo=tpe.suggest, max_evals=50)

        test_score = w.compute(best_config, test=True, name=f"{sampler} - {size}")

        print('Best found configuration:', best_config)
        
        print(f"Best config test score: {test_score}")

        results.append({
            **entry, 
            **test_score, 
            **best_config
        })

        df = pd.DataFrame(results)
        print(df.iloc[[-1]])
        df.to_csv(results_file, index=False)

plt.figure(figsize=(7, 6))
ax = plt.gca()
for d in roc_displays:
  d.plot(ax=ax, alpha=0.8)
plt.savefig('rocs.svg')

plt.figure(figsize=(7, 6))
ax = plt.gca()
for d in prc_displays:
  d.plot(ax=ax, alpha=0.8)
  
plt.legend( loc = 'upper right')
plt.savefig('prcs.svg')

