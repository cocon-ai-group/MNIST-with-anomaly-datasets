# -*- coding: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import numpy as np
import glob
from PIL import Image, ImageDraw
import os
import sys
import random


# ニューラルネットワークのモデル
class NMIST_Manifold_NN(chainer.Chain):

	def __init__(self):
		super(NMIST_Manifold_NN, self).__init__()
		with self.init_scope():
			self.layer1 = L.Linear(28*28, 100)
			self.layer2 = L.Linear(100, 50)
			self.layer3 = L.Linear(50, 20)
			self.layer4 = L.Linear(20, 2)
			self.layer5 = L.Linear(2, 20)
			self.layer6 = L.Linear(20, 50)
			self.layer7 = L.Linear(50, 100)
			self.layer8 = L.Linear(100, 28*28)

	def __call__(self, x):
		# ニューラルネットワークによる画像認識
		x = F.tanh(self.layer1(x))
		x = F.tanh(self.layer2(x))
		x = F.tanh(self.layer3(x))
		x = F.tanh(self.layer4(x))
		x = F.tanh(self.layer5(x))
		x = F.tanh(self.layer6(x))
		x = F.tanh(self.layer7(x))
		return self.layer8(x)

# ニューラルネットワークを作成
net = NMIST_Manifold_NN()
fun = F.mean_absolute_error
model = L.Classifier(net, lossfun=fun, accfun=fun)

# 全てのpngファイルを読み込む
train = []
for fn in glob.glob('train/*/*.png'):
	img = Image.open(fn).convert('L')
	x = np.array(img, dtype=np.float32)
	y = np.array(img, dtype=np.float32).reshape((28*28,))
	train.append((x, y))

# 繰り返し条件を作成する
train_iter = iterators.SerialIterator(train, 1000, shuffle=True)

# 学習アルゴリズムの選択
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)
updater = training.StandardUpdater(train_iter, optimizer, device=-1)
# 1000エポック分学習させる
trainer = training.Trainer(updater, (1000, 'epoch'), out="result")
# 学習の進展を表示するようにする
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch','main/loss']))
# 機械学習を実行する
trainer.run()

# 全てのpngファイルを読み込む
test = []
test_fn = []
for fn in glob.glob('anomaly/*.png'):
	img = Image.open(fn).convert('L')
	x = np.array(img, dtype=np.float32)
	x = x.reshape((1, 28, 28))  ##　畳み込みニューラルネットワークの場合
	test.append(x)
	test_fn.append(fn.split('/')[1])
test = np.array(test)
# ニューラルネットワークを一度だけ実行
result = net.manifold(test)
# 結果を保存
im = Image.new('RGB', (1000,1000), (0xff,0xff,0xff))
draw = ImageDraw.Draw(im)
xmax = np.max(result.data[:,0])
ymax = np.max(result.data[:,1])
xmin = np.min(result.data[:,0])
ymin = np.min(result.data[:,1])
for i in range(len(result.data)):
	l = test_fn[i]
	x = int((result.data[i][0]-xmin) / (xmax-xmin) * 900 + 50)
	y = int((result.data[i][1]-ymin) / (ymax-ymin) * 900 + 50)
	draw.text((x, y), l, (0,0,0))
im.save('ae.png', 'PNG')
