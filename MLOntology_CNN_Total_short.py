
# coding: utf-8

# In[1]:


import gensim
import os
import collections
import smart_open
import random
import multiprocessing
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pprint import pprint
import numpy as np
from sklearn import svm


# In[2]:


#global variabls

directory_path = "/home/hao/AnacondaProjects/MLOntology/"
data_path = directory_path + "data/"
vector_model_path = directory_path +"vectorModel/"
cnn_model_path = directory_path +"cnnModel/"


# In[3]:


conceptLabelDict={}
errors=[]

def read_label(fname):
    with smart_open.smart_open(fname) as f:
        for i, line in enumerate(f):
            #get the id for each concept paragraph
            splitted = line.decode("iso-8859-1").split("\t")
            if len(splitted)==3:
                conceptLabelDict[splitted[1]] = splitted[2].replace("\r\n", "")
            else:
                errors.append(splitted)

label_file = data_path + "ontClassLabels_july2017.txt"
read_label(label_file)


# In[4]:


conceptPairDict={}
errors=[]
conceptPairList=[]

def read_pair(fname):
    with smart_open.smart_open(fname) as f:
        for i, line in enumerate(f):
            #get the id for each concept paragraph
            splitted = line.decode("iso-8859-1").split("\t")
            if len(splitted)==3:
                conceptPairList.append([splitted[1], splitted[2].replace("\r\n", ""), 1])
#                 conceptPairDict[splitted[1]] = splitted[2].replace("\r\n", "")
            else:
                errors.append(splitted)

pair_file = data_path + "ontHierarchy_july2017.txt"
read_pair(pair_file)

first2pairs = conceptPairList[10:15]
print(first2pairs)
print(len(conceptPairList))


# In[5]:


conceptNotPairDict={}
conceptNotPairList=[]

def read_not_pair(fname):
    with smart_open.smart_open(fname) as f:
        for i, line in enumerate(f):
            #get the id for each concept paragraph
            splitted = line.decode("iso-8859-1").split("\t")
            if len(splitted)==2:
                conceptNotPairList.append([splitted[0], splitted[1].replace("\r\n", ""), 0])
#                 conceptNotPairDict[splitted[1]] = splitted[2].replace("\r\n", "")
            else:
                errors.append(splitted)

notPair_file = data_path + "taxNotPairs_july2017.txt"
read_not_pair(notPair_file)

# first2pairs = {k: conceptNotPairDict[k] for k in list(conceptNotPairDict)[10:15]}
first2pairs =conceptNotPairList[10:15]
print(first2pairs)
print(len(conceptNotPairList))

# In-place shuffle
random.shuffle(conceptNotPairList)
conceptNotPairList = conceptNotPairList[:len(conceptPairList)]

print(len(conceptNotPairList))


# In[6]:


vector_model_file = vector_model_path + "model0"

vector_model = gensim.models.Doc2Vec.load(vector_model_file)

inferred_vector = vector_model.infer_vector(['congenital', 'prolong', 'rupture', 'premature', 'membrane', 'lung'])
pprint(vector_model.docvecs.most_similar([inferred_vector], topn=10))


# In[7]:


# path = "D:/MLOntology/model1"

# model = gensim.models.Doc2Vec.load(path)

# inferred_vector = model.infer_vector(['congenital', 'prolong', 'rupture', 'premature', 'membrane', 'lung'])
# pprint(model.docvecs.most_similar([inferred_vector], topn=10))


# In[8]:


feature_number = 1024

def readFromModel(id_pair_list, id_notPair_list, model):
    pair_list = id_pair_list + id_notPair_list
    random.shuffle(pair_list)
    ids_list = []
    vector_list =[]
    label_list =[]
    for i, line in enumerate(pair_list):
        if line[0] in model.docvecs and line[1] in model.docvecs:
            a= model.docvecs[line[0]]
            b= model.docvecs[line[1]]
            c = np.array((a, b))
            ids_list.append((line[0], line[1]))
    #         test_list.append(np.reshape(c, feature_number))
            vector_list.append(np.reshape(c, feature_number, order='F'))
            label_list.append(line[2])
    return ids_list, vector_list, label_list

ids_list, vector_list, label_list= readFromModel(conceptPairList, conceptNotPairList, vector_model)

print(label_list[:20])


# In[9]:


del conceptPairList
del conceptNotPairList
del vector_model
import gc
gc.collect()


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(vector_list, label_list, ids_list, test_size = 0.2)
print(X_train[:20])
print(X_train[:20])
print(y_train[:20])
print(y_test[:20])
print(ids_test[:20])
print(ids_test[:20])


# In[11]:


#CNN
import numpy as np
import math
from math import sqrt

import tensorflow as tf


'''
In the data, there are 2 classes and every sample has 512 features
'''
# DATA_DIR = ''
CLASS_NUM = 2       #there are 2 classes
FEATURE_NUM = 1024   
TRAIN_ITER = 200    #the number of iterations for training
display_step = 100        #how many iterations to display the results
train_batch_size = 100


train_feature = np.asarray(X_train)      #training features (list of list)
train_y = y_train        #training lables    (list)
test_feature = np.asarray(X_test)       #test features  (list of list)
test_y = y_test         #test labels    (list)


y_m = np.eye(2)[train_y]
test_y_m = np.eye(2)[test_y]

'''
y = wx+b        (vectors)
'''
#function to get variables 'w'
def weight_variable(shape, num):
    initial = tf.truncated_normal(shape, stddev=1/num)
    return tf.Variable(initial, name='weight')

#the bias 'b' in the equations
def bias_variable(shape, num):
    initial = tf.constant(0.0001, shape=shape)
    return tf.Variable(initial, name='bias')

#convolutional process
def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride=1, padding='SAME')     #x: variable, w: weight, stride and padding (padding can be ignored currently) 

#pooling process
def max_pool_1x1(x, shape):
    x=tf.reshape(x,shape)       #it is transfered into four dimensions, but the other three are 1
    return tf.nn.max_pool(x, ksize=[1, 1, 4, 1],
                        strides=[1, 1, 2, 1], padding='SAME')

'''
The feature is 3 dimensional data.  [batch, length, channel] 
batch is usually ignored (for example there are 100 samples in a batch, so samples should not be modified mutually), length and channel are shown in the paper.
At first, the length is 512, and channel is 1.
Because our data are time series data, so length is enough, but for images, it may be [batch, length, width, channel]
'''
# the convolutional layer
def layer(features, f, input_n, channel, hidden_units, layer_index):
    """Construct a convolutional layer
    Args:
    features: Features placeholder, from the previous layer.
    f: the length
    input_n: Size of the features used in the convention.
    hidden_units: Size of the current hidden layer.
    layer_index: the index of layer
    Returns:
    hidden units: The unit output for the next layer.
    weights: the weights in the current hidden layer
    """
    # Hidden 1
    with tf.name_scope('hidden'+str(layer_index)) as scope:     # name scope may be ignored first
        with tf.name_scope("weight"):
            weights = weight_variable([input_n, channel, hidden_units], math.sqrt(f))

        with tf.name_scope("bias"):
            biases = bias_variable([hidden_units], math.sqrt(f))
    hidden = relu(conv1d(features, weights) + biases, 0.01)
    shape = [-1,1,f,hidden_units]
    h_pool1 = max_pool_1x1(hidden,shape)
    return h_pool1, weights

# fully connected layer, here the data are two dimension, [batch, length]
def densely_connect(features, input_n, hidden_units):
    """Construct a fully (densely) connected layer.
    Args:
    features: Features placeholder, from the previous layer.
    input_n: Size of units in the previous layer.
    hidden_units: Size of the current hidden layer.
    Returns:
    logits: The estimated output in last layer.
    weights: the weights in the hidden layer
    """
    with tf.name_scope('softmax_linear') as dense:
        with tf.name_scope("weight"):
            weights = weight_variable([input_n, hidden_units], math.sqrt(input_n))
        with tf.name_scope("bias"):
            biases = bias_variable([hidden_units], math.sqrt(input_n))
    logits = relu(tf.matmul(features, weights) + biases, 0.01)      # the matrix product operation
    return logits, weights

# dropout layer (it is not necessary)
# randomly set (1-keep_prob) percentage of units to be zero
def dropout(features, input_n, hidden_units, keep_prob):
    with tf.name_scope('dropout'):
        with tf.name_scope("weight"):
            weights = weight_variable([input_n, hidden_units], math.sqrt(input_n))
        with tf.name_scope("bias"):
            biases = bias_variable([hidden_units], math.sqrt(input_n))
    h_fc1_drop = tf.nn.dropout(features, keep_prob)
    drop_out = relu(tf.matmul(features, weights) + biases, 0.01)
    return drop_out

# calculate the loss in the neural network
def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size, NUM_CLASSES].
    Returns:
    loss: Loss tensor of type float.
    """
    with tf.name_scope("Loss"):
        labels = tf.to_int64(labels)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')
    # tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def next_batch(data, label, num):
    """Generate the next batch randomly
    Args:
    data: training data.
    label: training label.
    num: the size in a batch
    Returns:
    next batch's training features and labels.
    """
    index = np.arange(len(data))
    np.random.shuffle(index)
#     train_feature = data[np.array(index)[0:num]]
#     train_label = label[np.array(index)[0:num]]
#     return train_feature, train_label
    train_feature_batch = [data[b] for b in index[0:num]]
    train_feature_batch = np.asarray(train_feature_batch)
    train_label_batch = [label[b] for b in index[0:num]]
    train_label_batch = np.asarray(train_label_batch)
    return train_feature_batch, train_label_batch

def relu(x, alpha=0., max_value=None):
    '''ReLU.
    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x

#define a session to run the model
sess = tf.InteractiveSession()

#place holders for training features and label
#None means the value is variable
x = tf.placeholder(tf.float32, shape=[None, FEATURE_NUM], name ="input_vector")
y_ = tf.placeholder(tf.float32, shape=[None, CLASS_NUM], name = "class_label")

# decide whether it is training or testing, it is not used in our model, but it may be used
is_training = tf.placeholder(tf.bool)

#from [-1, 512, 1] -> [-1, 256, 32] -> [-1, 128, 64] -> [-1, 64, 64] -> [-1, 32, 64] -> [-1, 16, 64] -> [-1, 8, 64] -> [-1, 200]

#6 hidden layers
x_1 = tf.reshape(x, [-1,FEATURE_NUM,1])
h_pool0, w0 = layer(x_1, FEATURE_NUM, 15, 1, 32, 0)
h_pool0 = tf.reshape(h_pool0, [-1,512,32])
h_pool1, w1 = layer(h_pool0, 512, 10, 32, 64, 1)
h_pool1 = tf.reshape(h_pool1, [-1,256,64])
h_pool2, w2 = layer(h_pool1, 256, 10, 64, 64, 2)
h_pool2 = tf.reshape(h_pool2, [-1,128,64])
h_pool3, w3 = layer(h_pool2, 128, 10, 64, 64, 3)
h_pool3 = tf.reshape(h_pool3, [-1,64,64])
h_pool4, w4 = layer(h_pool3, 64, 5, 64, 64, 4)
h_pool4 = tf.reshape(h_pool4, [-1,32,64])
h_pool5, w5 = layer(h_pool4, 32, 5, 64, 64, 5)
h_pool5 = tf.reshape(h_pool5, [-1,16,64])
h_pool6, w6 = layer(h_pool5, 16, 5, 64, 64, 6)
h_pool6 = tf.reshape(h_pool6, [-1,8,64])

#densely connected: 200 units
h_pool_flat = tf.reshape(h_pool6, [-1, 8*64])
h_dc, w_d = densely_connect(h_pool_flat, 8*64, 200)

#dropout
keep_prob = tf.placeholder(tf.float32)
y_conv=dropout(h_dc, (int)(h_dc.get_shape()[1]), CLASS_NUM, keep_prob)


beta = 0.001
cross_entropy = loss(y_conv, y_)
loss = cross_entropy +beta*(tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1)+tf.nn.l2_loss(w2)+tf.nn.l2_loss(w3)+tf.nn.l2_loss(w4)+tf.nn.l2_loss(w5)+tf.nn.l2_loss(w6)+tf.nn.l2_loss(w_d))  #L2 regularization
epsilon = 1e-5      # learning rate
train_step = tf.train.AdamOptimizer(epsilon).minimize(loss)     #optimization function, our goal is to minimize the loss

predict = tf.argmax(y_conv,1, name ="predict")   #the predicted class

# calculate the accuray, the corrected classified divided by the total size
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1), name="correct_prediction")
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")

#saver to save the training check point
# variables can be restored in a new model by 'saver.restore(sess, save_path)'
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())  #initialize the variables


for i in range(1,TRAIN_ITER):       #training iterations
    d, l = next_batch(train_feature, y_m, train_batch_size)      # get batch_size samples in one batch
#     print("d size is %s, l size is %s "% (len(d), len(l)))
#     print("d size is %s, l size is %s "% (len(d[0]), len(l[0])))
    _, ls=sess.run([train_step,cross_entropy], feed_dict={x: d, y_: l, keep_prob: 1, is_training:True})     #run the train step (optimization function), the second one is just to show the loss in this iteration.   THE FEED dictionary is to feed the place holders which are needed in the optimization function.
    
    if i%display_step==0:
        print(_, i)
        acc = sess.run([accuracy], feed_dict={x: d, y_: l, keep_prob: 1, is_training:False})
        print("Train Loss:", ls, "Acc:", acc)


# In[13]:


# save the model results
save_path = saver.save(sess, cnn_model_path + "/model-noleaky.ckpt")
print("Model saved in file: %s" % save_path)

print("done")


# In[17]:


# sess.run  or tensor.eval are two ways
# get the accuracy in the testing data
# need to cut down the size of testing data into batches

print("feature len ", len(test_feature))
print("label len ", len(test_y_m))
test_batch_size = 1024
print("%s iterations in total" % (len(test_feature)//test_batch_size))
for i in range(len(test_feature)//test_batch_size):
    test_feature_batch = test_feature[ i*test_batch_size : min(i*test_batch_size +test_batch_size, len(test_feature))]
    test_y_m_batch = test_y_m[ i*test_batch_size : min(i*test_batch_size +test_batch_size,len(test_feature))]
    print("%d iteration accurarcy: %s" % (i, accuracy.eval(session=sess, feed_dict={x:test_feature_batch, y_:test_y_m_batch,keep_prob: 1, is_training:False})))

    
    


# In[ ]:


y_pred = sess.run(predict, feed_dict={x:test_feature_batch, keep_prob:1, is_training:False})
err_ids=np.flatnonzero(np.eye(2)[y_pred] != test_y_m_batch)
for err_id in err_ids:
    print("index %d predicted label %s, but true label is %s" % (err_id, y_pred[err_id], test_y_m_batch[err_id]))
    idpair = ids_test[err_id] 
    concept1 = conceptLabelDict[idpair[0]]
    concept2 = conceptLabelDict[idpair[1]]
    print("%s Concept Pairs: (%s --- %s)" % (idpair, concept1, concept2 ))



# In[ ]:


result = y_pred
test_label_list = test_y

from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
print(accuracy_score(result, test_label_list))
print(average_precision_score(result, test_label_list))

print(f1_score(result, test_label_list, average='macro') ) 

print(f1_score(result, test_label_list, average='micro')  )

print(f1_score(result, test_label_list, average='weighted') )

print(f1_score(result, test_label_list, average=None))

print(precision_score(result, test_label_list, average=None))
print(recall_score(result, test_label_list, average=None))

print(roc_auc_score(result, test_label_list, average=None))

