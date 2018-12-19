import pandas as pd
import numpy as np
import mxnet as mx
import logging
from mxnet.lr_scheduler import MultiFactorScheduler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# data = pd.read_csv('./data/data1218.csv')
# numpy_matrix = data.as_matrix()
# numpy_matrix = numpy_matrix.astype('float32')
# np.random.shuffle(numpy_matrix)
# numpy_matrix.shape

# train_data = numpy_matrix[:80000, 1:-1]
# train_label = numpy_matrix[:80000, -1].astype('int32')
# print(train_data.shape)
# print(train_label.shape)
# test_data = numpy_matrix[80000:, 1:-1]
# test_label = numpy_matrix[80000:, -1].astype('int32')
# print(test_data.shape)
# print(test_label.shape)

data = pd.read_csv('./data/processed.csv')
label = data.readmitted.as_matrix()
data = data.drop(columns=['Unnamed: 0', 'readmitted']).as_matrix()

train_data, test_data, train_label, test_label = train_test_split(
    data, label, test_size=0.2, random_state=0
)
train_data, train_label = SMOTE(random_state=0).fit_sample(train_data, train_label)

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

logging.getLogger().setLevel(logging.DEBUG)

def NaiveNet():
    data = mx.symbol.var(
        name='data'
    )
    out = mx.symbol.FullyConnected(
        data=data, 
        num_hidden=1024, 
        no_bias=True, 
        name='FC1'
    )
    out = mx.symbol.BatchNorm(
        data=out,
        eps=1e-5,
        momentum=0.9
    )
    # out = mx.symbol.Dropout(
    #     data=out, 
    #     p=0.5
    # )
    out = mx.symbol.Activation(
        data=out, 
        act_type='relu'
    )
    out = mx.symbol.FullyConnected(
        data=out, 
        num_hidden=1024,
        no_bias=True,
        name='FC2'
    )
    out = mx.symbol.BatchNorm(
        data=out,
        eps=1e-5,
        momentum=0.9
    )
    # out = mx.symbol.Dropout(
    #     data=out, 
    #     p=0.5
    # )
    out = mx.symbol.Activation(
        data=out, 
        act_type='relu'
    )
    out = mx.symbol.FullyConnected(
        data=out, 
        num_hidden=3, 
        name='Out'
    )
    out = mx.symbol.SoftmaxOutput(
        data=out, 
        name='softmax'
    )
    return out

NNet = NaiveNet()

Model = mx.mod.Module(
    symbol=NNet,
    context=mx.cpu()
)

class F1Metric(mx.metric.EvalMetric):
    def __init__(self):
        super(F1Metric, self).__init__('F1-Metric')
    def update(self, labels, preds):
        pred = preds[0].asnumpy()
        label = labels[0].asnumpy()
        # print(pred.shape)
        pass

base_lr = 0.1
lr_factor = 10
lr_step = '10, 13'
lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
begin_epoch = 0
lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
lr_iters = [int(epoch * len(train_data) / batch_size) for epoch in lr_epoch_diff]
print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)

lr_scheduler = MultiFactorScheduler(
    lr_iters, 
    1.0 / lr_factor
)

Model.fit(
    train_iter, 
    eval_data=test_iter, 
    optimizer='sgd', 
    optimizer_params={
        'learning_rate': 0.1,
        # 'wd': 0.001,
        'lr_scheduler': lr_scheduler
    },
    eval_metric='acc',
    batch_end_callback=mx.callback.Speedometer(batch_size, 10),
    num_epoch=15
)

Prob = Model.predict(test_iter).asnumpy()
Prob = Prob.argmax(axis=1)
print(Prob.shape)
print(f1_score(test_label.asnumpy(), Prob, average='macro'))