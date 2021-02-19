from kge.graph import KGraph

import kge
from kge.models import TransE, DistMult, Rescal, ComplEx
from kge.loss_functions import LogisticLoss, PairwiseHingeLoss, Regularized, NLLMulticlass
from kge.negative_samplers import UniformNegativeSamplerFast
from kge.utils import timeit, strip_whitespace

import torch
from torch.optim import SGD, SparseAdam, Adam

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

# losses = [NLLMulticlass(), PairwiseHingeLoss(margin=1), LogisticLoss(), PairwiseHingeLoss(margin=.5)]

objectives = []

# for d in [100, 50, 150]:
#     for loss in losses:
#         if not isinstance(loss, LogisticLoss):
#             objectives.append((TransE(d, p=2), loss))

#         objectives.append((DistMult(d), loss))
#         objectives.append((Rescal(d), loss))
#         objectives.append((ComplEx(d), loss))

#         for l_e, l_r in [(0.01, 0.01), (0.1, 0.1)]:

#             if not isinstance(loss, LogisticLoss):
#                 objectives.append((TransE(d, p=2, normalize_embeddings=False), Regularized(loss, l_e, l_r)))

#             objectives.append((DistMult(d, normalize_embeddings=False), Regularized(loss, l_e, l_r)))
#             objectives.append((Rescal(d, normalize_embeddings=False), Regularized(loss, l_e, l_r)))
#             objectives.append((ComplEx(d, normalize_embeddings=False), Regularized(loss, l_e, l_r)))

# objectives = [(TransE(200, normalize_embeddings=True, p=2), PairwiseHingeLoss(margin=.5))]
objectives = [(Rescal(50, normalize_embeddings=False), Regularized(PairwiseHingeLoss(margin=.5), 0.01, 0.01))]

print('# objectives: ', len(objectives))

n_negatives = [30]

optimizer = Adam

learning_rate = 0.000501

max_batch_size = 2500

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

results_path = 'results/results_v1.csv'

results = []
if os.path.isfile(results_path):
    results = pd.read_csv(results_path).to_dict('records')

for dataset, files in datasets.items():

    print(dataset)
    g_train, g_valid, g_test = KGraph.from_csv(files, columns=[0,2,1])

    print(g_train)
    g_train.add_inverse_triples()
    print(g_train)

    for n_epochs in [100, 200, 300, 500]:

        for objective in objectives:
            
            model, loss = copy.deepcopy(objective)

            for n_neg in n_negatives:

                negative_sampler = UniformNegativeSamplerFast(n_neg)

                for i in range(5):

                    batch_size = max_batch_size // (2 ** i)

                    entry = {
                        'Dataset': dataset, 
                        'Model': str(model), 
                        'Loss': str(loss), 
                        'Negatives': n_neg,
                        'Optimizer': optimizer.__name__,
                        'Learning Rate': learning_rate,
                        'Batch Size': batch_size,
                        'Epochs': n_epochs
                    }

                    if results and any(all(result.get(key) == val for key, val in entry.items()) for result in results):
                        print('skipping')
                        break

                    negative_sampler.init(g_train, device)

                    model.init(g_train, device)

                    loss.init(g_train, device)

                    try: 
                        with timeit("Training"):
                            kge.train(
                                g_train,
                                model,
                                loss,
                                negative_sampler=negative_sampler,
                                optimizer=optimizer(list(model.parameters()), lr=learning_rate),
                                n_epochs=n_epochs,
                                device=device,
                                batch_size=batch_size,
                                verbose=True,
                                checkpoint=True,
                                checkpoint_dir='checkpoints',
                                checkpoint_period=10
                            )
                    except RuntimeError as e:
                        if 'CUDA' in str(e):
                            print(e)
                            continue
                        else:
                            raise e


                    with timeit("Evaluation"):
                        stats, _, _ = kge.evaluate(model, g_test, device, g_train, batch_size=10 * batch_size)

                    entry = {**entry, **stats}
                    results.append(entry)

                    df = pd.DataFrame(results)
                    print(df.iloc[[-1]])
                    df.to_csv(results_path, index=False)
                    break
                
            del model, loss