#-*- coding: utf-8 -*- 
import os,sys
import random
import find_mxnet
import mxnet as mx
import string
import math
import cv2
import glob

import numpy as np
import cPickle as p
from PIL import Image,ImageDraw,ImageFont

from cnn_predict import cnn_predict

BATCH_SIZE = 1

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
	self.label = label
	self.data_names = data_names
	self.label_names = label_names
         
	self.pad = 0
	self.index = None
    @property
    def provide_data(self):
        return [(n,x.shape) for n,x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n,x.shape) for n,x in zip(self.label_names, self.label)]

class LRCNIter(mx.io.DataIter):
    def __init__(self, dataset, labelset, num, batch_size, seq_len, init_states):
        
	self.batch_size = batch_size
	self.count = (num-seq_len)/batch_size
	self.seq_len = seq_len
	self.dataset = dataset
	self.labelset = labelset
	
	self.init_states = init_states
	self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

	self.provide_data = [('data',(batch_size, seq_len, 2048))]+init_states
	self.provide_label = [('label',(batch_size, seq_len, ))]

    def __iter__(self):
        init_state_names = [x[0] for x in self.init_states]
	for k in range(self.count):
	    data = []
	    label = []
	    for i in range(self.batch_size):
	        data_seq = []
		label_seq = []
		idx = k * self.batch_size + i
		#tmp = random.randint(0, self.listset[idx]-self.seq_len)
		for j in range(self.seq_len):
	            idx_1 = idx + j
		    data_seq.append(self.dataset[idx_1])
		    label_seq.append(self.labelset[idx_1])
		data.append(data_seq)
		label.append(label_seq)
	
	    data_all = [mx.nd.array(data)]+self.init_state_arrays
	    label_all = [mx.nd.array(label)]
	    data_names = ['data']+init_state_names
	    label_names = ['label']

	    data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
	    yield data_batch

    def reset(self):
        pass

if __name__ == '__main__':
#def predict():
    num_hidden = 2048
    num_lstm_layer = 2
    batch_size = BATCH_SIZE

    num_epoch = 400
    learning_rate = 0.0025
    momentum = 0.0015
    num_label = 5
    seq_len = 10
    
    devs = [mx.context.gpu(0)]

    init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
    init_states = init_c + init_h

   #f_2 = file('test_data.data')
   #x_test = p.load(f_2)

   #f_3 = file('test_label.data')
   #y_test = p.load(f_3)

    (x_test, y_test) = cnn_predict()
    data_test = LRCNIter(x_test, y_test, len(x_test), batch_size, seq_len, init_states)
    print data_test.provide_data,data_test.provide_label
    
    #model = mx.model.FeedForward.load("./model/lstm", epoch=num_epoch, ctx=devs, num_batch_size=BATCH_SIZE)
    model = mx.model.FeedForward.load("./model/lstm", epoch=400, ctx=devs)

    internels = model.symbol.get_internals()
    #print internels.list_outputs()
    #print model.arg_params
    #fea_symbol = internels['softmax_output']
    fea_symbol = internels['fc_output']
    feature_exactor = mx.model.FeedForward(ctx=devs, symbol=fea_symbol, num_batch_size=1, 
                                           arg_params=model.arg_params, aux_params=model.aux_params,
					   allow_extra_params=True)

    cnn_test_result = feature_exactor.predict(data_test)
    predict_result = []
    print np.array(cnn_test_result).shape
    #print len(cnn_test_result[0])
    for i in range(len(cnn_test_result)):
        predict_result.append(np.argmax(cnn_test_result[i]))

    print predict_result
    #return predict_result
    pic = []
    for filename in glob.glob('./1/image*.jpg'):
        pic.append(filename)
    pic.sort()

    for i in range(len(pic)-10):
        idx = i+10
	dic = {0:'no', 1:'stable', 2:'wave', 3:'Positive rotation', 4:'Reverse rotation'}
        ttfont = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Bold.ttf",20) 
        im = Image.open(pic[idx])
        draw = ImageDraw.Draw(im)
        draw.text((10,10),dic[predict_result[i]], fill=(0,0,0),font=ttfont)
        im.save(pic[idx])
	#draw.text((40,40),unicode('杨利伟','utf-8'), fill=(0,0,0),font=ttfont)


