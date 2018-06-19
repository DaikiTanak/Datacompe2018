from feature_extractor import Feature_Extractor
from model import MLP

import chainer
from chainer.datasets import split_dataset_random
from chainer import iterators
import chainer.links as L
import chainer.functions as F
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from chainer import serializers
import progressbar

ext = Feature_Extractor()
X_test = ext.X_test


model = L.Classifier(MLP())
serializers.load_npz("NN.model", model)
gpu_id = 0
if gpu_id >= 0:
    chainer.cuda.get_device(gpu_id).use()
    model.to_gpu(gpu_id)

y_pred = []
bar = progressbar.ProgressBar()
print("start predicting...")
for x in bar(X_test, max_value=len(X_test)):
    x = model.xp.asarray(x[None, ...])
    y = model.predictor(x)
    result = F.softmax(y, axis=1)
    pred = result[0][1].data
    y_pred.append(pred)
    


f = open('submit.dat', 'w') # 書き込みモードで開く
y_pred = list(map(str, y_pred))
for p in y_pred:
    f.writelines(p + "\n") # シーケンスが引数。
f.close()
