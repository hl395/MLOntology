
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


# In[2]:


#read class label file
#create mapping from id to labels  
#iso-8859-1 , encoding="iso-8859-1"
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

label_file = "ontClassLabels.txt"
read_label(label_file)
print(conceptLabelDict["446087008"])
print(conceptLabelDict["132818006"])
print(errors)


# In[17]:


print(len(conceptLabelDict))


# In[3]:


conceptMappingDict={}


# In[18]:


def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname) as f:
        for i, line in enumerate(f):
            #get the id for each concept paragraph
            splitted = line.decode("iso-8859-1").split("\t", 1)
            
            line = line.decode("iso-8859-1")
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                conceptMappingDict[i]= int(splitted[0])
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])



train_file = "ontClassTopology.txt"

train_corpus = list(read_corpus(train_file))


# In[20]:


print(len(train_corpus))
train_corpus[11256:11257]




# In[21]:


cores = multiprocessing.cpu_count()

print(cores)
models = [
    # PV-DBOW 
    Doc2Vec(dm=0, dbow_words=1, size=200, window=8, min_count=1, iter=10, workers=cores),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=200, window=8, min_count=1, iter =10, workers=cores),
]


# In[22]:


models[0].build_vocab(train_corpus)
print(str(models[0]))
models[1].reset_from(models[0])
print(str(models[1]))


# In[ ]:


for model in models:
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)


# In[ ]:

for model in models:
	# inferred_vector = model.infer_vector(['clinical', 'finding', 'evaluation', 'prevent', 'sampling', 'foot'])
	inferred_vector = model.infer_vector(['congenital', 'prolong', 'rupture', 'premature', 'membrane', 'lung'])
	sims = model.docvecs.most_similar([inferred_vector], topn=10)
	pprint(sims)


from tempfile import mkstemp

for model in models:
    fs, temp_path = mkstemp("gensim_temp" + model.dm)  # creates a temp file
    model.save(temp_path)





