from itertools import groupby
from operator import itemgetter

import torch
from torch.optim import Adam

import numpy
import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

from kge.graph import KGraph
from kge.models import TransE, DistMult, Rescal, ComplEx
from kge.loss_functions import LogisticLoss, PairwiseHingeLoss, Regularized, NLLMulticlass
from kge.negative_samplers import UniformNegativeSamplerFast
from kge.training import train
from kge.evaluation import evaluate

from datasets import datasets


class KGEWorker(Worker):

    def __init__(self, model_cls, train_graph, validation_graph, nameserver='127.0.0.1'):
        super().__init__(nameserver=nameserver, run_id='kge')
        self.model_cls = model_cls
        self.train_graph = train_graph
        self.validation_graph = validation_graph


    def compute(self, config, budget, **kwargs):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        loss = eval(config['loss'])()

        if 'margin' in config:
            loss.margin = config['margin']

        

        model = self.model_cls(config['embedding_dim'])
            
        if config['regularization'] == 'Lp':
            model.normalize_embeddings = False
            loss = Regularized(loss, config['reg_coeff'], p=config['p_reg'])

        if self.model_cls == TransE:
            model.p = config['p_transe']

        model.init(self.train_graph, device)
        loss.init(self.train_graph, device)

        neg_sampler = UniformNegativeSamplerFast(config['n_negatives'])
        neg_sampler.init(self.train_graph, device)

        train(
            graph=self.train_graph,
            model=model,
            criterion=loss,
            negative_sampler=neg_sampler,
            optimizer=Adam(model.parameters(), config['learning_rate']),
            n_epochs=budget,
            device=device,
            batch_size=config['batch_size'],
            use_checkpoint=True,
            checkpoint_dir='checkpoints',
            checkpoint_period=10,
            verbose=True
        )

        scores = evaluate(model, self.validation_graph, device, verbose=True)

        return { 'loss': -scores['both']['mrr'], 'info': scores }


    def get_configspace(self):
        config_space = CS.ConfigurationSpace(seed=0)
        
        embeding_dim = CSH.UniformIntegerHyperparameter('embedding_dim', lower=50, upper=300, default_value=100, q=10)
        n_negatives = CSH.UniformIntegerHyperparameter('n_negatives', lower=2, upper=100, default_value=10, q=2)
        loss = CSH.CategoricalHyperparameter('loss', choices=['PairwiseHingeLoss', 'LogisticLoss', 'NLLMulticlass'])
        regularization = CSH.CategoricalHyperparameter('regularization', choices=['normalization', 'Lp'])
        learning_rate = CSH.UniformFloatHyperparameter('learning_rate', lower=1e-5, upper=1e-2, default_value=1e-3, log=True)
        batch_size = CSH.CategoricalHyperparameter('batch_size', choices=[1000, 5000])
        
        margin = CSH.CategoricalHyperparameter('margin', [0.5, 1])
        reg_coeff = CSH.UniformFloatHyperparameter('reg_coeff', lower=1e-5, upper=0.1, log=True)
        p_reg = CSH.CategoricalHyperparameter('p_reg', choices=[1, 2, 3], default_value=2)
        
        config_space.add_hyperparameters([embeding_dim, n_negatives, loss, regularization, learning_rate, batch_size, margin, reg_coeff, p_reg])

        if self.model_cls == TransE:
            p_transe = CSH.CategoricalHyperparameter('p_transe', choices=[1, 2], default_value=2)
            config_space.add_hyperparameter(p_transe)
        #     config_space.add_forbidden_clause(CS.ForbiddenAndConjunction(
        #         CS.ForbiddenEqualsClause(p_transe, 1),
        #         CS.ForbiddenEqualsClause(loss, 'PairwiseHingeLoss')
        #     ))

        config_space.add_conditions([
            CS.EqualsCondition(margin, loss, 'PairwiseHingeLoss'),
            CS.EqualsCondition(reg_coeff, regularization, 'Lp'),
            CS.EqualsCondition(p_reg, regularization, 'Lp')
        ])

        return config_space


result_logger = hpres.json_result_logger(directory='.', overwrite=True)

NS = hpns.NameServer(run_id='kge', host='127.0.0.1', port=None)
NS.start()


for dataset, files in datasets.items():
    print(dataset)
    g_train, g_valid, g_test = KGraph.from_csv(files, columns=[0,2,1])

    w = KGEWorker(TransE, g_train, g_valid)
    w.run(background=True)

    bohb = BOHB(
        configspace=w.get_configspace(),
        run_id='kge',
        nameserver='127.0.0.1',
        result_logger=result_logger,
        eta=2,
        min_budget=50,
        max_budget=800
    )
    res = bohb.run(n_iterations=3)

    bohb.shutdown(shutdown_workers=True)

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/args.max_budget))

NS.shutdown()