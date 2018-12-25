import numpy as np
import mxnet as mx
import pandas as pd

import logging

from mxnet.lr_scheduler import MultiFactorScheduler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from Model import NaiveNet

logging.getLogger().setLevel(logging.DEBUG)

data = pd.read_csv('./data/processed.csv')
label = data.readmitted.as_matrix()
data = data.drop(columns=['Unnamed: 0', 'readmitted']).as_matrix()

random_state = 0

def EpochCheck(mod, test_iter, test_label):
    def _callback(iter_no, sym=None, arg=None, aux=None):
        Prob = mod.predict(test_iter).asnumpy()
        Prob = Prob.argmax(axis=1)
        print('[Epoch %d]'%(iter_no))
        print(classification_report(y_true=test_label.asnumpy(), y_pred=Prob, digits=4))
    return _callback

SKFold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

def TrainLoop(train_iter, test_iter, test_label):
    NNet = NaiveNet()
    Model = mx.mod.Module(
        symbol=NNet,
        context=mx.cpu()
    )

    base_lr = 0.01
    lr_factor = 0.1
    lr_step = '10'
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    begin_epoch = 0
    end_epoch = 20
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(train_data) / batch_size) for epoch in lr_epoch_diff]
    print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)
    lr_scheduler = MultiFactorScheduler(
        lr_iters,
        lr_factor
    )

    Model.fit(
        train_iter,
        eval_data=test_iter,
        optimizer='sgd',
        optimizer_params={
            'learning_rate': base_lr,
            'wd': 0.0001,
            'lr_scheduler': lr_scheduler
        },
        eval_metric='acc',
        # eval_metric=mx.metric.F1(average='macro'),
        batch_end_callback=mx.callback.Speedometer(batch_size, 200),
        epoch_end_callback=[EpochCheck(Model, test_iter, test_label)],
        num_epoch=end_epoch
    )

    Prob = Model.predict(test_iter).asnumpy()
    Prob = Prob.argmax(axis=1)

    return classification_report(y_true=test_label.asnumpy(), y_pred=Prob, digits=4)

KFoldIdx = 0
KFoldResult = []
for train_idx, test_idx in SKFold.split(data, label):
    KFoldIdx += 1
    train_data, test_data = data[train_idx], data[test_idx]
    train_label, test_label = label[train_idx], label[test_idx]
    train_data, train_label = SMOTE(random_state=random_state).fit_sample(train_data, train_label)

    train_data = mx.nd.array(train_data)
    train_label = mx.nd.array(train_label)
    test_data = mx.nd.array(test_data)
    test_label = mx.nd.array(test_label)

    batch_size = 100

    train_iter = mx.io.NDArrayIter(
        data=train_data,
        label=train_label,
        batch_size=batch_size,
        shuffle=True
    )
    test_iter = mx.io.NDArrayIter(
        data=test_data,
        label=test_label,
        batch_size=batch_size
    )

    print('K-Fold %d'%(KFoldIdx))
    KFoldResult.append(TrainLoop(train_iter, test_iter, test_label))

for Item in KFoldResult:
    print(Item)








# Random-State
# [0] : 0.376988
# [1] : 0.388603
# [2] : 0.381658
# [3] : 0.383682
# [4] : 0.376800
