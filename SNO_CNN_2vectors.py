
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
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
import time


# In[2]:

iterations_list = [20] 
for iterations in iterations_list:
    print('Run training with iter: ', str(iterations))    
    #global variabls
    
    directory_path =  "/home/h/hl395/mlontology/SNO/"
    #data_path = directory_path + "data/hierarchy/"
    data_path = directory_path + "data/"
    #data_path = directory_path + "data/parea/"
    vector_model_path = directory_path +"vectorModel/" + str(iterations) + "/"
    cnn_model_path = directory_path +"cnnModel/2vectors/"+ str(iterations) + "/"
    img_path = directory_path + "img/"
    
       
    img_name = "sno_2vectors_iter_"+ str(iterations) + "_"
    # In[3]:
    
    
    import re
    def get_trailing_number(s):
        m = re.search(r'\d+$', s)
        return m.group() if m else None
    
    
    # In[4]:
    
    
    #read class label file
    #create mapping from id to labels  
    #iso-8859-1 , encoding="iso-8859-1"
    
    def read_label(fname):
        conceptLabelDict = {}
        with smart_open.smart_open(fname) as f:
            for i, line in enumerate(f):
              #get the id for each concept paragraph
              splitted = line.decode("iso-8859-1").split("\t")
              if len(splitted)==3:
                  conceptID = get_trailing_number(splitted[1])
                  conceptLabelDict[conceptID] = splitted[2].replace("\r\n", "")
              else:
                  errors.append(splitted)
        return conceptLabelDict
    
    label_file_2017 = data_path + "ontClassLabels_july2017.txt"
    conceptLabelDict_2017=read_label(label_file_2017)
    label_file_2018= data_path + "ontClassLabels_jan2018.txt"
    conceptLabelDict_2018=read_label(label_file_2018)
    
    
    # read positive samples
    # In[5]:
    
    
    conceptPairDict={}
    errors=[]
    conceptPairList=[]
    
    def read_pair(fname):
        with smart_open.smart_open(fname) as f:
            for i, line in enumerate(f):
                #get the id for each concept paragraph
                splitted = line.decode("iso-8859-1").split("\t")
                if len(splitted)==3:
                    childID = get_trailing_number(splitted[1])
                    parentID = get_trailing_number(splitted[2].replace("\r\n", ""))
                    conceptPairList.append([childID, parentID , 1])
    #                 conceptPairDict[splitted[1]] = splitted[2].replace("\r\n", "")
                else:
                    errors.append(splitted)
    
    pair_file = data_path + "ontHierarchy_july2017.txt"
    read_pair(pair_file)
    
    checkpairs = conceptPairList[10:15]
    print(checkpairs)
    print("number of pairs: ", len(conceptPairList))
    
    
    # read negative samples
    
    # In[7]:
    
    conceptNotPairDict={}
    conceptNotPairList=[]
    
    def read_not_pair(fname):
        with smart_open.smart_open(fname) as f:
            for i, line in enumerate(f):
                #get the id for each concept paragraph
                splitted = line.decode("iso-8859-1").split("\t")
                if len(splitted)==2:
                    childID = get_trailing_number(splitted[0])
                    notparentID = get_trailing_number(splitted[1].replace("\r\n", ""))
                    conceptNotPairList.append([childID, notparentID, 0])
    #                 conceptNotPairDict[splitted[1]] = splitted[2].replace("\r\n", "")
                else:
                    errors.append(splitted)
    
    notPair_file = data_path + "taxNotPairs_sno.txt"
    #notPair_file = data_path + "taxNotPairs_sno_all.txt"
    #notPair_file = data_path + "taxNotPairs_sno_parent_grandparent_sibling.txt"    
    read_not_pair(notPair_file)
    
    checkNonpairs =conceptNotPairList[10:15]
    print(checkNonpairs)
    print("number of not pairs: ", len(conceptNotPairList))
    
    # remove duplicates
    cleanlist = []
    [cleanlist.append(x) for x in conceptNotPairList if x not in cleanlist]
    print("After remove duplicates in not linked pairs: ")
    print(len(cleanlist))
    conceptNotPairList = cleanlist
     
    # In[8]:
    
    
    if len(conceptPairList) < len(conceptNotPairList):
        # Downsampling negative samples
        random.shuffle(conceptNotPairList)
        leftPairList = conceptNotPairList[len(conceptPairList):]
        conceptNotPairList = conceptNotPairList[:len(conceptPairList)]
    else:
        # Upsampling negative samples
        random.shuffle(conceptNotPairList)
        duplicatedList = conceptNotPairList[:len(conceptPairList) - len(conceptNotPairList)]
        print(len(duplicatedList))
        random.shuffle(conceptNotPairList)
        conceptNotPairList.extend(duplicatedList)
    
    assert len(conceptPairList) == len(conceptNotPairList), "Mistmatch in Positive & Negative samples!"
    
    
    # In[ ]:
    
    
    #  PV-DBOW
    vector_model_file_0 = vector_model_path + "model0"
    
    vector_model_0 = gensim.models.Doc2Vec.load(vector_model_file_0)
    
    inferred_vector_0= vector_model_0.infer_vector(['congenital', 'prolong', 'rupture', 'premature', 'membrane', 'lung'])
    pprint(vector_model_0.docvecs.most_similar([inferred_vector_0], topn=10))
    
    # In[9]:
    
    
    # PV-DM seems better??
    vector_model_file = vector_model_path + "model1"
    
    vector_model = gensim.models.Doc2Vec.load(vector_model_file)
    
    inferred_vector = vector_model.infer_vector(['congenital', 'prolong', 'rupture', 'premature', 'membrane', 'lung'])
    pprint(vector_model.docvecs.most_similar([inferred_vector], topn=10))
    
    
    # In[10]:
    
    
    #vector_model.docvecs['7918']
    
    
    # read both negative and positive pairs into pairs list and label list
    # In[11]:
    
    def readFromPairList(id_pair_list, id_notPair_list):
        pair_list = id_pair_list + id_notPair_list
        random.shuffle(pair_list)
        idpairs_list =[]
        label_list =[]
        for i, line in enumerate(pair_list):      
            idpairs_list.append([line[0], line[1]])
            label_list.append(line[2])
        return idpairs_list, label_list
    
    idpairs_list, label_list= readFromPairList(conceptPairList, conceptNotPairList)
    
    print(label_list[:20])
    
    
    # split samples into training and validation set
    # In[12]:
    
    from sklearn.model_selection import train_test_split
    X_train, X_validation, y_train, y_validation = train_test_split(idpairs_list, label_list, test_size = 0.2, shuffle= True)
    print(X_train[:20])
    print(X_validation[:20])
    print(y_train[:20])
    print(y_validation[:20])
    
    
    # get vector for each concept
    # In[13]:
    
    
    def getVectorFromModel(concept_id, conceptLabelDict, model, opt_str=""):
        if concept_id in model.docvecs:
            concept_vector= model.docvecs[concept_id]
        else:
            print("%s not found, get inferred vector "%(concept_id))
            concept_label = conceptLabelDict[concept_id]
            concept_vector= model.infer_vector(op_str.split() + concept_label.split())
        return concept_vector
    
    def getVector(line, conceptLabelDict, model, opt_str=""):        
        a = getVectorFromModel(line[0], conceptLabelDict, model, opt_str)
        b = getVectorFromModel(line[1], conceptLabelDict, model, opt_str)
        c = np.array((a, b))
        c = c.T 
    #     c = np.expand_dims(c, axis=2)
    #     print(c.shape)
        return c
    
    
    # stack vectors into 4 channels
    # In[14]:
    
    def stackVector(vector1, vector2):
        return np.concatenate((vector1, vector2),axis=1)
    
    
    # batch generator
    # In[15]:
    
    n_classes=2 
    
    def get_batches(x_samples, y_samples, conceptLabelDict, batch_size=64):
        samples = list(zip(x_samples, y_samples))
        num_samples = len(samples)
        
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
    
            X_samples = []
            Y_samples= []
            for batch_sample in batch_samples:
                pair_list = batch_sample[0]
    #            data_vector = getVector(pair_list, conceptLabelDict, vector_model)
                pvdm_vector = getVector(pair_list, conceptLabelDict, vector_model)
                pvdbow_vector = getVector(pair_list, conceptLabelDict, vector_model_0)
                data_vector = stackVector(pvdm_vector, pvdbow_vector)
    #                 data_vector = stackVector(data_vector)
    #                 print(data_vector.shape)
                X_samples.append(data_vector)
                class_label = batch_sample[1] 
                Y_samples.append(class_label)
    
            X_samples = np.array(X_samples).astype('float32')
            Y_samples = np.eye(n_classes)[Y_samples]
    #             print('one batch ready')
            yield shuffle(X_samples, Y_samples)
    
    
    # def get_batches(X, y, batch_size = 100):
    #     """ Return a generator for batches """
    #     n_batches = len(X) // batch_size
    #     X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]
    # 
    #     # Loop over batches and yield
    #     for b in range(0, len(X), batch_size):
    #         yield X[b:b+batch_size], y[b:b+batch_size]
    
    # In[16]:
    
    
    # Imports
    import tensorflow as tf
    
    # build the model??
    batch_size = 1000       # Batch size : 2000
    seq_len = 512          # word embedding length
    learning_rate = 0.0001
    lambda_loss_amount = 0.001
    epochs = 20  # 2000
    
    n_classes = 2
    n_channels = 4
    
    
    graph = tf.Graph()
    
    # Construct placeholders
    with graph.as_default():
        inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels], name = 'inputs')
        labels_ = tf.placeholder(tf.float32, [None, n_classes], name = 'labels')
        keep_prob_ = tf.placeholder(tf.float32, name = 'keep')
        learning_rate_ = tf.placeholder(tf.float32, name = 'learning_rate')
    
        
    with graph.as_default():
        # (batch, 512, 4) --> (batch, 256, 18)
        conv1 = tf.layers.conv1d(inputs=inputs_, filters=18, kernel_size=2, strides=1, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')
    
        # (batch, 256, 18) --> (batch, 128, 36)
        conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=36, kernel_size=2, strides=1, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
    
        # (batch, 128, 36) --> (batch, 64, 72)
        conv3 = tf.layers.conv1d(inputs=max_pool_2, filters=72, kernel_size=2, strides=1, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=2, strides=2, padding='same')
        
        # (batch, 64, 72) --> (batch, 32, 144)
        conv4 = tf.layers.conv1d(inputs=max_pool_3, filters=144, kernel_size=2, strides=1, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_4 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')
        
        # (batch, 32, 144) --> (batch, 16, 144)  # 288
        conv5 = tf.layers.conv1d(inputs=max_pool_4, filters=144, kernel_size=2, strides=1, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_5 = tf.layers.max_pooling1d(inputs=conv5, pool_size=2, strides=2, padding='same')
        
        # (batch, 16, 144) --> (batch, 8, 144)   #576
        conv6 = tf.layers.conv1d(inputs=max_pool_5, filters=144, kernel_size=2, strides=1, 
                                 padding='same', activation = tf.nn.relu)
        max_pool_6 = tf.layers.max_pooling1d(inputs=conv6, pool_size=2, strides=2, padding='same')
    
    
    with graph.as_default():
        # Flatten and add dropout
        flat = tf.reshape(max_pool_6, (-1, 8*144))
        flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
        
        # Predictions
        logits = tf.layers.dense(flat, n_classes, name='logits')
        predict = tf.argmax(logits,1, name ="predict")   #the predicted class
        probability = tf.nn.softmax(logits, name ="probability")		
        
        # L2 loss prevents this overkill neural network to overfit the data
        l2 = lambda_loss_amount * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()) 
            
        # Cost function and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_))+l2
        optimizer = tf.train.AdamOptimizer(learning_rate_).minimize(cost)
        
        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_, 1), name='correct_pred')
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    
    
    
    # In[17]:
    
    
    if (os.path.exists(cnn_model_path) == False):
        os.makedirs(cnn_model_path)
    
    
    # In[ ]:
    
    
    validation_acc = []
    validation_loss = []
    
    train_acc = []
    train_loss = []
    
    with graph.as_default():
        saver = tf.train.Saver()
    
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 1
       
        training_start_time = time.time()
        # Loop over epochs
        for e in range(epochs):
            
            # Loop over batches
            for x,y in get_batches(X_train, y_train, conceptLabelDict_2017, batch_size):
                
                # Feed dictionary
                feed = {inputs_ : x, labels_ : y, keep_prob_ : 0.5, learning_rate_ : learning_rate}
                
                # Loss
                loss, _ , acc = sess.run([cost, optimizer, accuracy], feed_dict = feed)
                train_acc.append(acc)
                train_loss.append(loss)
                
                # Print at each 50 iters
                if (iteration % 50 == 0):
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {:d}".format(iteration),
                          "Train loss: {:6f}".format(loss),
                          "Train acc: {:.6f}".format(acc)
                          )
                
                # Compute validation loss at every 100 iterations
                if (iteration%100 == 0):                
                    val_acc_ = []
                    val_loss_ = []
                    
                    for x_v, y_v in get_batches(X_validation, y_validation, batch_size):
                        # Feed
                        feed = {inputs_ : x_v, labels_ : y_v, keep_prob_ : 1.0}  
                        
                        # Loss
                        loss_v, acc_v = sess.run([cost, accuracy], feed_dict = feed)                    
                        val_acc_.append(acc_v)
                        val_loss_.append(loss_v)
                    
                    # Print info
                    print("Epoch: {}/{}".format(e, epochs),
                          "Iteration: {:d}".format(iteration),
                          "Validation loss: {:6f}".format(np.mean(val_loss_)),
                          "Validation acc: {:.6f}".format(np.mean(val_acc_)))
                    
                    # Store
                    validation_acc.append(np.mean(val_acc_))
                    validation_loss.append(np.mean(val_loss_))
                
                # Iterate 
                iteration += 1
        training_duration = time.time() - training_start_time
        print("Total training time: {}".format(training_duration))
        saver.save(sess, cnn_model_path + "har.ckpt")
    
    
    # In[ ]:
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Plot training and test loss
    t = np.arange(iteration-1)
    
    plt.figure(figsize = (8,6))
    plt.plot(t, np.array(train_loss), 'r-', t[t % 100 == 0], np.array(validation_loss), 'b*')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(img_path + img_name + 'loss.png') 
    plt.show()
    
    
    # In[ ]:
    
    
    # Plot Accuracies
    plt.figure(figsize = (8,6))
    plt.plot(t, np.array(train_acc), 'r-', t[t % 100 == 0], validation_acc, 'b*')
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    #plt.ylim(0.4, 1.0)
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(img_path + img_name + 'accuracy.png') 
    plt.show()
    
    
    # In[ ]:
    
    
    # test new data 
    print('\n\nTesting with new data')
    
    import json
    from pprint import pprint
  
    jsonFile = data_path + "2018newconcepts.json"
  
    test_data = json.load(open(jsonFile))

  # pprint(data)

  # pprint(data['100841000119109']['Ancestors'][1])


  # In[5]:


    def readFromJsonData2(test_data):
      result_paired = []
      result_not_paired= []
      for key, value in test_data.items():
        if value['Parents']:
            for x in range(len(value['Parents'])):
              if value['Parents'][x] in conceptLabelDict_2017:
                result_paired.append([key, value['Parents'][x], 1])
        if value['Siblings']:
            for x in range(len(value['Siblings'])):
              if value['Siblings'][x] in conceptLabelDict_2017:
                result_not_paired.append([key, value['Siblings'][x], 0])
        if value['Children']:
            for x in range(len(value['Children'])):
              if value['Children'][x] in conceptLabelDict_2017:
                result_not_paired.append([key, value['Children'][x], 0])
        return result_paired, result_not_paired
      
      
    def readFromJsonData(test_data):
      result_paired = []
      result_not_paired= []
      for key, value in test_data.items():
        if value['Parents']:
          for x in range(len(value['Parents'])):
            result_paired.append([key, value['Parents'][x], 1])
        if value['Siblings']:
          for x in range(len(value['Siblings'])):
            result_not_paired.append([key, value['Siblings'][x], 0])
        #if value['Children']:
        #  for x in range(len(value['Children'])):
        #    result_not_paired.append([key, value['Children'][x], 0])
      return result_paired, result_not_paired




  # In[6]:


    paired_list, not_paired_list = readFromJsonData(test_data)
    pprint(paired_list[:20])
    pprint(not_paired_list[:20])
    
    # remove duplicates
    cleanlist = []
    [cleanlist.append(x) for x in paired_list if x not in cleanlist]
    print("After remove duplicates in paired pairs: ")
    print(len(cleanlist))
    paired_list = cleanlist

  # remove duplicates
    cleanlist = []
    [cleanlist.append(x) for x in not_paired_list if x not in cleanlist]
    print("After remove duplicates in not paired pairs: ")
    print(len(cleanlist))
    not_paired_list = cleanlist
    
    
    

    def processConceptParents(json_data):
      parents_dict = {}
      for key, value in json_data.items():
        parents_list = []
        if value['Parents']:
          for x in range(len(value['Parents'])):
            parents_list.append(value['Parents'][x])
        parents_dict[key]= parents_list   
      return parents_dict

    def readOutPairsTwo(key, parent, paired_list, not_paired_list):
      pair_list=[]
      not_pair_list=[]
      for i in range(len(paired_list)):
        pair=paired_list[i]
        if pair[0] == key and pair[1]!= parent:
          pair_list.append(pair)
      for i in range(len(not_paired_list)):
        pair=not_paired_list[i]
        if pair[0] == key:
          not_pair_list.append(pair)
      return pair_list, not_pair_list
    
    
    n_classes = 2
    def get_batches_NotRandom(x_samples, y_samples, conceptLabelDict, batch_size=64, op_str=''):
        samples = list(zip(x_samples, y_samples))
        num_samples = len(samples)
        
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
    
            X_samples = []
            Y_samples= []
            for batch_sample in batch_samples:
                pair_list = batch_sample[0]
    #            data_vector = getVector(pair_list, conceptLabelDict, vector_model)
                pvdm_vector = getVector(pair_list, conceptLabelDict, vector_model, op_str)
                pvdbow_vector = getVector(pair_list, conceptLabelDict, vector_model_0, op_str)
                data_vector = stackVector(pvdm_vector, pvdbow_vector)
    #                 data_vector = stackVector(data_vector)
    #                 print(data_vector.shape)
                X_samples.append(data_vector)
                class_label = batch_sample[1] 
                Y_samples.append(class_label)
    
            X_samples = np.array(X_samples).astype('float32')
            Y_samples = np.eye(n_classes)[Y_samples]
    #             print('one batch ready')
            yield X_samples, Y_samples
    
    
    


    parents_dict = processConceptParents(test_data)
  
  
    test_rest_acc =[]
    true_label_list=[]
    predicted_label_list=[]
    predicted_prob=[]
    test_id_lists= []
  
    for key, parents in parents_dict.items():
      print("Processing key ", key)
      if len(parents) > 1:
        for i in range(len(parents)):            
          parent = parents[i]
          print("\tProcessing its %d parent %s "%(i, parent))
          if parent in conceptLabelDict_2017:
            parent_str = conceptLabelDict_2017[parent]
          else:
            parent_str = conceptLabelDict_2018[parent]
          pair_list, not_pair_list = readOutPairsTwo(key, parent, paired_list, not_paired_list)	
          if pair_list:
            idpairs_list, label_list= readFromPairList(pair_list, not_pair_list)
            test_id_lists.extend(idpairs_list)
            for x_t, y_t in get_batches_NotRandom(idpairs_list, label_list, conceptLabelDict_2018, len(idpairs_list), parent_str):
              feed = {inputs_: x_t,
                labels_: y_t,
                keep_prob_: 1}
              
              true_label_list.extend(label_list)
              
              batch_acc = sess.run(accuracy, feed_dict=feed)
              test_rest_acc.append(batch_acc)
              
              label_pred = sess.run(predict, feed_dict=feed)
              predicted_label_list.extend(label_pred)
              
              pred_prob = sess.run(probability, feed_dict=feed)
              predicted_prob.extend(pred_prob[:,1])
              print("\t\t Predict: ", label_pred)
              print("\t\t True label: ", label_list)
  
    print("Test accuracy: {:.6f}".format(np.mean(test_rest_acc)))	
  
    from sklearn.metrics import classification_report
    print(classification_report(true_label_list, predicted_label_list))			


    
    # Now we're going to assess the quality of the neural net using ROC curve and AUC
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    
    # send the actual dependent variable classifications for param 1, 
    # and the confidences of the true classification for param 2.
    FPR, TPR, _ = roc_curve(true_label_list, predicted_prob)
    
    # Calculate the area under the confidence ROC curve.
    # This area is equated with the probability that the classifier will rank 
    # a randomly selected defaulter higher than a randomly selected non-defaulter.
    AUC = auc(FPR, TPR)
    
    # What is "good" can dependm but an AUC of 0.7+ is generally regarded as good, 
    # and 0.8+ is generally regarded as being excellent 
    print("AUC is {}".format(AUC))
    
    # Now we'll plot the confidence ROC curve 
    plt.figure()
    plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(img_path + img_name + 'roc.png') 
    plt.show()
    
    
    
    negative_sample_error_list=[]
    for i,x in enumerate(true_label_list):
        if x!= predicted_label_list[i] and x == 0:
            negative_sample_error_list.append(test_id_lists[i])
    print(len(negative_sample_error_list))
    
    
    def get_label_from_id(id):
      if id in conceptLabelDict_2017:
        return conceptLabelDict_2017[id]
      elif id in conceptLabelDict_2018:
        return conceptLabelDict_2018[id]
      else:
        print("{} not exists in dictionary".format(id)) 
    

    for batch_sample in negative_sample_error_list:
        print("{} : {} ".format(batch_sample[0], batch_sample[1]))
        print("{} -> {} ".format(get_label_from_id(batch_sample[0]), get_label_from_id(batch_sample[1])))
    
    


print("testing done")