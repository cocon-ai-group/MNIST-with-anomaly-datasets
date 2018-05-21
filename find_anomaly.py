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

# 全てのpngファイルを読み込む
train = []
train_fn = []
for fn in glob.glob('train/*/*.png'):
	img = Image.open(fn).convert('L')
	x = np.array(img, dtype=np.float32)
	x = x.reshape((1, 28, 28))  ##　畳み込みニューラルネットワークの場合
	train.append(x)
	train_fn.append(fn.split('/')[1])
train = np.array(train)

# Metric学習のデータを取得する関数
triplet_pos = 0
# 一枚画像を取得
def get_one():
	global triplet_pos
	data = train[triplet_pos]
	triplet_pos = triplet_pos+1
	if triplet_pos >= len(train):
		triplet_pos = 0
	return data

# 1トリプレットを取得
def get_one_triple():
	if random.random() < 0.5:
		c = get_one()
		d = np.zeros(c.shape, dtype=np.float32)
		e = np.zeros(c.shape, dtype=np.float32) + 255
	else:
		d = get_one()
		e = get_one()
		c = np.zeros(d.shape, dtype=np.float32)
		if random.random() < 0.5:
			c = c + 255
	return (c,d,e)
	
# ニューラルネットワークのモデル
class NMIST_Triplet_NN(chainer.Chain):

	def __init__(self):
		super(NMIST_Triplet_NN, self).__init__()
		with self.init_scope():
			self.layer1 = L.Linear(28*28, 50)
			self.layer2 = L.Linear(50, 50)
			self.layer3 = L.Linear(50, 50)
			self.layer4 = L.Linear(50, 2)

	def __call__(self, x):
		# ニューラルネットワークによるMetric認識
		x = F.tanh(self.layer1(x))
		x = F.tanh(self.layer2(x))
		x = F.tanh(self.layer3(x))
		return self.layer4(x)

# カスタムUpdaterのクラス
class TripletUpdater(training.StandardUpdater):

	def __init__(self, optimizer, device):
		self.loss_val = []
		super(TripletUpdater, self).__init__(
			None,
			optimizer,
			device=device
		)

	# イテレーターがNoneなのでエラーが出ないようにオーバライドする
	@property
	def epoch(self):
		return 0

	@property
	def epoch_detail(self):
		return 0.0

	@property
	def previous_epoch_detail(self):
		return 0.0

	@property
	def is_new_epoch(self):
		return False
		
	def finalize(self):
		pass
	
	def update_core(self):
		batch_size = 1000
		# Optimizerを取得
		optimizer = self.get_optimizer('main')
		# Tripletを取得
		anchor = []
		positive = []
		negative = []
		for i in range(batch_size):
			in_data = get_one_triple()
			anchor.append(in_data[0])
			positive.append(in_data[1])
			negative.append(in_data[2])
		anchor = np.array(anchor)
		positive = np.array(positive)
		negative = np.array(negative)
		# ニューラルネットワークを3回実行
		model = optimizer.target
		anchor_r = model(anchor)
		positive_r = model(positive)
		negative_r = model(negative)
		# Triplet Lossで学習
		optimizer.update(F.triplet, anchor_r, positive_r, negative_r)

# ニューラルネットワークを作成
model = NMIST_Triplet_NN()
# Optimizerの作成
optimizer = optimizers.Adam()
optimizer.setup(model)
# デバイスを選択してTrainerを作成する
updater = TripletUpdater(optimizer, device=-1)
trainer = training.Trainer(updater, (2000, 'iteration'), out="result")
# 学習の進展を表示するようにする
trainer.extend(extensions.ProgressBar(update_interval=1))
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
result = model(test)
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
im.save('triplet.png', 'PNG')

# クラスタリング
from sklearn import cluster
clf = cluster.AgglomerativeClustering(n_clusters=2, linkage='average', affinity='l2')
clz = clf.fit_predict(result.data)
# クラスタ番号が0か1なので、0以外の数を数えて、クラスタの大きさを比較
count1 = np.count_nonzero(clz)
count2 = len(clz) - count1
# 小さい方のクラスタを取得
clzidx = 1 if count1 < count2 else 0
# 小さい方のクラスタに属しているインデックスを取得
idx = np.argwhere(clz==clzidx)[:,0]

# 色分けして保存
im2 = Image.new('RGB', (1000,1000), (0xff,0xff,0xff))
draw = ImageDraw.Draw(im2)
for i in range(len(result.data)):
	l = test_fn[i]
	c = (0xff,0,0) if clz[i] == clzidx else (0x80,0x80,0x80)
	x = int((result.data[i][0]-xmin) / (xmax-xmin) * 900 + 50)
	y = int((result.data[i][1]-ymin) / (ymax-ymin) * 900 + 50)
	draw.text((x, y), l, c)
im2.save('clusters.png', 'PNG')

# 小さい方のクラスタに属したPNGファイルを取得
import shutil
for i in idx:
	print(test_fn[i]) # ファイル名
	shutil.copyfile('anomaly/%s'%test_fn[i], test_fn[i])

