
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
    conceptLabelDict={}
    with smart_open.smart_open(fname) as f:
        for i, line in enumerate(f):
            #get the id for each concept paragraph
            splitted = line.decode("iso-8859-1").split("\t")
            if len(splitted)==3:
                conceptLabelDict[splitted[1]] = splitted[2].replace("\r\n", "")
            else:
                errors.append(splitted)
    return conceptLabelDict

label_file_2017 = data_path + "ontClassLabels_july2017.txt"
conceptLabelDict_2017=read_label(label_file_2017)
label_file_2018= data_path + "ontClassLabels_jan2018.txt"
conceptLabelDict_2018=read_label(label_file_2018)


# In[4]:


import json
from pprint import pprint

jsonFile = data_path + "2018newconcepts.json"

data = json.load(open(jsonFile))

# pprint(data)

# pprint(data['100841000119109']['Ancestors'][1])


# In[5]:


def readFromJsonData(data):
    result_paired = []
    result_not_paired= []
    for key, value in data.items():
        if value['Parents']:
            for x in range(len(value['Parents'])):
                result_paired.append([key, value['Parents'][x], 1])
        if value['Siblings']:
            for x in range(len(value['Siblings'])):
                result_not_paired.append([key, value['Siblings'][x], 0])
        if value['Children']:
            for x in range(len(value['Children'])):
                result_not_paired.append([key, value['Children'][x], 0])
    return result_paired, result_not_paired
    
    


# In[6]:


paired_list, not_paired_list = readFromJsonData(data)
pprint(paired_list[:20])
pprint(not_paired_list[:20])


# In[7]:


vector_model_file = vector_model_path + "model0"

vector_model = gensim.models.Doc2Vec.load(vector_model_file)

inferred_vector = vector_model.infer_vector(['congenital', 'prolong', 'rupture', 'premature', 'membrane', 'lung'])
pprint(vector_model.docvecs.most_similar([inferred_vector], topn=10))


# In[8]:


def getInferredVector(concept_id, conceptLabelDict, model):
    concept_label = conceptLabelDict[concept_id]
    concept_vector= model.infer_vector(concept_label.split())
    return concept_vector


# In[9]:


feature_number = 1024

def getInferredVectorFromModel(id_pair_list, id_notPair_list, conceptLabelDict, model):
    pair_list = id_pair_list + id_notPair_list
    random.shuffle(pair_list)
    list_ids = {}
    vector_list =[]
    label_list =[]
    for i, line in enumerate(pair_list):        
        a= getInferredVector(line[0], conceptLabelDict, model)
        b= getInferredVector(line[1], conceptLabelDict, model)
        c = np.array((a, b))
        list_ids[i] = (line[0], line[1])
#         test_list.append(np.reshape(c, feature_number))
        vector_list.append(np.reshape(c, feature_number, order='F'))
        label_list.append(line[2])
    return list_ids, vector_list, label_list

list_ids, vector_list, label_list= getInferredVectorFromModel(paired_list, not_paired_list, conceptLabelDict_2018, vector_model)

print(vector_list[:20])
print(label_list[:20])


# In[11]:


import tensorflow as tf
 
sess=tf.Session()    
#First let's load meta graph and restore weights
cnn_model_file = cnn_model_path + 'model-noleaky.ckpt.meta'
saver = tf.train.import_meta_graph(cnn_model_file)
saver.restore(sess,tf.train.latest_checkpoint(cnn_model_path))


# In[12]:


graph = tf.get_default_graph()
x = graph.get_tensor_by_name("input_vector:0")
y = graph.get_tensor_by_name("class_label:0")
keep_prob = tf.placeholder(tf.float32)
predict= graph.get_tensor_by_name("predict:0")
correct_prediction= graph.get_tensor_by_name("correct_prediction:0")
accuracy= graph.get_tensor_by_name("accuracy:0")


# In[ ]:


# correct_prediction = tf.equal(predict, tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")


# In[ ]:


test_batch_size = 1024
print("%s iterations in total" % (len(vector_list)//test_batch_size))
for i in range(len(vector_list)//test_batch_size):
    vector_list_batch = vector_list[ i*test_batch_size : min(i*test_batch_size +test_batch_size, len(vector_list))]
    label_list_batch = label_list[ i*test_batch_size : min(i*test_batch_size +test_batch_size, len(label_list))]
    acc = sess.run(accuracy, feed_dict={x:vector_list_batch, y: np.eye(2)[label_list_batch], keep_prob:1})
    print("iteration %d Acc: %s" % (i,acc))


# In[ ]:


accuracy.eval(session=sess, feed_dict={x:test_feature_batch, y_:test_y_m_batch,keep_prob: 1, is_training:False})

