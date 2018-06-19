import pandas as pd
import numpy as np
from collections import Counter
import hashlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from collections import defaultdict

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.datasets import split_dataset_random
from chainer import iterators
from chainer import training
from chainer import serializers
chainer.config.use_cudnn = 'auto'
from chainer import datasets
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from chainer.training.triggers import MinValueTrigger
from chainer.datasets import tuple_dataset


from model import MLP
from feature_extractor import Feature_Extractor

ext = Feature_Extractor()
train, test = ext.train, ext.test

print(test)


""" training """
batchsize = 1024
train_iter = iterators.SerialIterator(train, batchsize, shuffle=True)
test_iter = iterators.SerialIterator(
    test, batchsize, repeat=False, shuffle=False)


model = L.Classifier(MLP())
gpu_id = 0
if gpu_id >= 0:
    chainer.cuda.get_device(gpu_id).use()
    model.to_gpu(gpu_id)

a=0.0001
optimizer = chainer.optimizers.Adam(alpha=a)
optimizer.setup(model)
updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)


max_epoch = 300
# TrainerにUpdaterを渡す
trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='result')

from chainer.training import extensions
trainer.extend(extensions.LogReport())
#trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id), name='val')
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.ProgressBar(update_interval=1))

def save_best_model(t):
    print("saving model..")
    serializers.save_npz("NN.model", model)

trainer.extend(save_best_model,
                trigger=MinValueTrigger('val/main/loss',
                trigger=(1, 'epoch')))


print("start training...")
trainer.run()
