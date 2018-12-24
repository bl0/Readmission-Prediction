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

random_state = 2

train_data, test_data, train_label, test_label = train_test_split(
    data, label, test_size=0.1, random_state=random_state
)
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

logging.getLogger().setLevel(logging.DEBUG)

def NaiveNet():
    data = mx.symbol.var(
        name='data'
    )
    out = mx.symbol.FullyConnected(
        data=data, 
        num_hidden=256,
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
        num_hidden=256,
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
        num_hidden=128,
        no_bias=True,
        name='FC3'
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
        num_hidden=128,
        no_bias=True,
        name='FC4'
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
        num_hidden=64,
        no_bias=True,
        name='FC5'
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

def EpochCheck(mod, test_iter, test_label):
    def _callback(iter_no, sym=None, arg=None, aux=None):
        Prob = mod.predict(test_iter).asnumpy()
        Prob = Prob.argmax(axis=1)
        print('[Epoch %d] F1 Score -> %.6f'%(iter_no, f1_score(test_label.asnumpy(), Prob, average='macro')))
    return _callback

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
    batch_end_callback=mx.callback.Speedometer(batch_size, 100),
    epoch_end_callback=[EpochCheck(Model, test_iter, test_label)],
    num_epoch=end_epoch
)

Prob = Model.predict(test_iter).asnumpy()
Prob = Prob.argmax(axis=1)
print(Prob.shape)
print('F1 Score -> %.6f'%(f1_score(test_label.asnumpy(), Prob, average='macro')))

# Random-State
# [0] : 0.376988
# [1] : 0.388603
# [2] : 0.381658
# [3] : 0.383682
# [4] : 0.376800
