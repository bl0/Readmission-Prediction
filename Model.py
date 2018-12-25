import mxnet as mx

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