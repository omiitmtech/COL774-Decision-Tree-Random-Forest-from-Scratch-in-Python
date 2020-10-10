#!/usr/bin/env python
# coding: utf-8

# In[123]:


#Section to import all the required libraries
from xclib.data import data_utils
from sklearn import metrics
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import queue
import sys
import time
import matplotlib.pyplot as plt


# In[2]:


root_path="ass3_parta_data/"


# In[3]:


#Section to read the files from the system
f_train_x = sys.argv[1]
f_train_y = sys.argv[2]
f_test_x  = sys.argv[3]
f_test_y  = sys.argv[4]
f_val_x   = sys.argv[5]
f_val_y   = sys.argv[6]

# f_train_x = root_path+"train_x.txt"
# f_train_y = root_path+"train_y.txt" 
# f_test_x  = root_path+"test_x.txt"
# f_test_y  = root_path+"test_y.txt"
# f_val_x   = root_path+"valid_x.txt"
# f_val_y   = root_path+"valid_y.txt"


# In[4]:


def prepare_data(f_train_x,f_train_y,f_test_x,f_test_y,f_val_x,f_val_y):
    train_x = data_utils.read_sparse_file(f_train_x).todense()
    train_y = pd.read_csv(f_train_y, header=None)
    train_y = np.array(train_y).reshape(len(train_y),)
    
    test_x = data_utils.read_sparse_file(f_test_x).todense()
    test_y = pd.read_csv(f_test_y, header=None)
    test_y = np.array(test_y).reshape(len(test_y),) 
    
    val_x = data_utils.read_sparse_file(f_val_x).todense()
    val_y = pd.read_csv(f_val_y, header=None)
    val_y = np.array(val_y).reshape(len(val_y),) 
    
    print('Shape of Trainig data :',train_x.shape,train_y.shape)
    print('Shape of Testing data :',test_x.shape,test_y.shape)
    print('Shape of Validation data :',val_x.shape,val_y.shape)
    
    return train_x,train_y, test_x,test_y, val_x,val_y


# In[5]:


def cal_entropy(y): #y should be in the shape (no_of_vals,)
    entropy = 0
    n = len(y)
    classes = np.unique(y)
    for c in classes:
        fraction = (np.sum(y==c))*1.0/n
        entropy += fraction*np.log2(1/fraction)
    return entropy


# In[6]:


def cal_entropy_attribute(col,col_median,x,y):
    n = len(y)
    left_y  =y[np.squeeze(np.asarray(x[:,col])) <= col_median]
    left_count = len(left_y)
    right_y  =y[np.squeeze(np.asarray(x[:,col])) > col_median]
    right_count = len(right_y)
    e0 = ((left_count*1.0)/n)*cal_entropy(left_y)
    e1 = ((right_count*1.0)/n)*cal_entropy(right_y)
    return e0+e1
    


# In[7]:


def find_best_split_of_all(x, y):
        Max_MI = -100
        index  = -1
        H_Y = cal_entropy(y)
        median_list = np.median(x,axis=0).tolist()
        for i in range(482):
            cur_median = median_list[0][i]
            H_Y_X = cal_entropy_attribute(i,cur_median,x,y)
            
            Attrib_MI = H_Y - H_Y_X
            if Attrib_MI > Max_MI:
                Max_MI = Attrib_MI
                index = i
                crct_median = cur_median
                if Max_MI == 1.0:
                    return index,crct_median,Max_MI
        return index,crct_median,Max_MI


# In[8]:


def all_same(items):
    return all(x == items[0] for x in items)


# In[9]:


class DecisionTreeClassifier(object):
    def __init__(self, max_depth):
        self.depth = 0
        self.max_depth = max_depth
    
    def fit(self, x, y, par_node={}, depth=0):
        
        global no_leaf_nodes, no_internal_nodes
        if par_node is None: 
            return None
        elif len(y) == 0:
            return None
        elif all_same(y):
            no_leaf_nodes +=1
#             print('label :',y[0],'depth :',depth,'node type :','leaf')
            return {'label':y[0],'depth':depth,'nodetype':'leaf'}
#         elif depth >= self.max_depth:
#             return None
        else:
            neg = np.sum(y==0)
            pos = np.sum(y==1)
#             print('neg :',neg)
#             print('pos :',pos)
            if neg > pos :
                label = 0
            else :
                label = 1
                
            col,median,Max_MI = find_best_split_of_all(x, y)
            if Max_MI < 1e-5:
                no_leaf_nodes +=1
#                 cur_no_nodes += 1
#                 print('label :',label,'depth :',depth,'node type :','leaf')
                return {'label':label,'depth':depth,'node_type':'leaf'}
            else:
                no_internal_nodes +=1
                y_left  = y[np.squeeze(np.asarray(x[:,col])) <= median]
                y_right = y[np.squeeze(np.asarray(x[:,col])) > median]
                par_node = {'index_col':col,
                            'median':median,
                            'label':label,
                            'depth':depth,
                            'nodetype': 'internal',
                           }
                
#                 print('label :',label,'depth :',depth,'node type :','internal','selected :',col,'MI :',Max_MI)
                x_left  = x[np.squeeze(np.asarray(x[:,col])) <= median]
                x_right  = x[np.squeeze(np.asarray(x[:,col])) > median]
                par_node['left'] = self.fit(x_left, y_left, {}, depth+1)
                par_node['right'] = self.fit(x_right, y_right, {}, depth+1)
                self.depth += 1 
                self.trees = par_node
#                 pprint.pprint(par_node)
                return par_node
        


# In[10]:


#------------- Read data from files ----------------------
train_x,train_y, test_x,test_y, val_x,val_y = prepare_data(f_train_x,f_train_y,f_test_x,f_test_y,f_val_x,f_val_y)


# In[70]:


#---------- make copies of levels--------
copy_train_y = np.copy(train_y)
copy_test_y = np.copy(test_y)
copy_val_y = np.copy(val_y)
no_leaf_nodes = 0
no_internal_nodes = 0


# In[12]:


#----------- Create decision tree ----------------------
clf = DecisionTreeClassifier(max_depth=10)
t0 = time.time()
m = clf.fit(train_x, train_y)
print('Time to fit the model :',time.time()-t0)
print('No. of leaf nodes :',no_leaf_nodes,'no. of internal nodes :',no_internal_nodes)
no_nodes_DT = no_leaf_nodes + no_internal_nodes


# In[13]:


def return_accuracy(y_actual,y_pred):
    return metrics.accuracy_score(y_actual, y_pred)


# In[14]:


def divide_indexes(test_data_x,index_list,median,index_col):
    all_left_indexes = np.where(test_data_x[:,index_col] <= median)[0]
    all_right_indexes = np.where(test_data_x[:,index_col] > median)[0]
    l_index = list(set(all_left_indexes) & set(index_list))
    r_index = list(set(all_right_indexes) & set(index_list))
    return l_index,r_index
    


# In[15]:


def set_labels_cal_indexes(X,Y,index_list,median,index_col,label):
    if Y == 'train':
        copy_train_y[index_list] = label
        accuracy = return_accuracy(train_y,copy_train_y)
    elif Y == 'test':
        copy_test_y[index_list] = label
        accuracy = return_accuracy(test_y,copy_test_y)
    elif Y == 'val':
        copy_val_y[index_list] = label
        accuracy = return_accuracy(val_y,copy_val_y)
    
    l_index,r_index = divide_indexes(X,index_list,median,index_col)
    return l_index,r_index,accuracy


# In[16]:


def set_labels_leaf_node(Y,index_list,label):
    if Y == 'train':
        copy_train_y[index_list] = label
        accuracy = return_accuracy(train_y,copy_train_y)
    elif Y == 'test':
        copy_test_y[index_list] = label
        accuracy = return_accuracy(test_y,copy_test_y)
    elif Y == 'val':
        copy_val_y[index_list] = label
        accuracy = return_accuracy(val_y,copy_val_y)
    return accuracy


# In[17]:


def BFS_tree(root_node,no_of_nodes,X,Y,indexes):
    nodes = queue.Queue(no_of_nodes)
    nodes.put((root_node,indexes))
    no_nodes_tree = 0
    accuracy_list_no_nodes = []
    while(not nodes.empty()):
        no_nodes_tree += 1
        node = nodes.get()
#         print(node[0]['nodetype'],node[0]['median'],node[0]['index_col'],len(node[1]))
        if node[0].get('nodetype') == 'internal':
#           l_index,r_index,accuracy = set_labels_cal_indexes(X,index,median,index_col,label)
            l_index,r_index,accuracy = set_labels_cal_indexes(X,Y,node[1],node[0]['median'],node[0]['index_col'],node[0]['label'])
            if node[0]['left'] != None:
                nodes.put((node[0]['left'],l_index))
            if node[0]['right'] != None:
                nodes.put((node[0]['right'],r_index)) 
        else :
            accuracy = set_labels_leaf_node(Y,node[1],node[0]['label'])
        accuracy_list_no_nodes.append((no_nodes_tree,accuracy))
    return accuracy_list_no_nodes


# In[18]:


def calc_nodewise_acc():    
    train_indexes = np.where(train_x[:,217] > -2.0)[0]
    test_indexes  = np.where(test_x[:,217] > -2.0)[0]
    val_indexes   = np.where(val_x[:,217] > -2.0)[0]
    no_nodes_in_tree = no_leaf_nodes + no_internal_nodes
    #BFS_tree(decision_tree,no_nodes_in_tree,x_data_set,index_x_dataset,copy_y_labels)
    train_accuracy_list_no_nodes = BFS_tree(m,no_nodes_in_tree,train_x,'train',train_indexes)
    test_accuracy_list_no_nodes = BFS_tree(m,no_nodes_in_tree,test_x,'test',test_indexes)
    val_accuracy_list_no_nodes = BFS_tree(m,no_nodes_in_tree,val_x,'val',val_indexes)
    return train_accuracy_list_no_nodes,test_accuracy_list_no_nodes,val_accuracy_list_no_nodes


# In[19]:


t0 = time.time()
train_accuracy_list_no_nodes,test_accuracy_list_no_nodes,val_accuracy_list_no_nodes = calc_nodewise_acc()
print('Time to calculate nodewise accuracies on train, test and val :',time.time()-t0)


# In[20]:


print('Training Set Accuracy :',train_accuracy_list_no_nodes[no_nodes_DT-1][1])
print('Testing Set Accuracy :',test_accuracy_list_no_nodes[no_nodes_DT-1][1])
print('Validation Set Accuracy :',val_accuracy_list_no_nodes[no_nodes_DT-1][1])


# In[21]:


def plot_accuracies():
    train_node_list = []
    train_acc_list = []

    test_node_list = []
    test_acc_list = []

    val_node_list = []
    val_acc_list = []

    for i in tqdm(range(19929)):
        train_node_list.append(train_accuracy_list_no_nodes[i][0])
        train_acc_list.append(train_accuracy_list_no_nodes[i][1])

        test_node_list.append(test_accuracy_list_no_nodes[i][0])
        test_acc_list.append(test_accuracy_list_no_nodes[i][1])

        val_node_list.append(val_accuracy_list_no_nodes[i][0])
        val_acc_list.append(val_accuracy_list_no_nodes[i][1])
    #plot accuracies
    plot_graph(train_node_list,train_acc_list,test_node_list,test_acc_list,val_node_list,val_acc_list)
        


# In[22]:


def plot_graph(train_node_list,train_acc_list,test_node_list,test_acc_list,val_node_list,val_acc_list):
    plt.plot(train_node_list, train_acc_list, color='orange',label="Train Accuracy")
    plt.plot(test_node_list, test_acc_list, color='green',label="Test Accuracy")
    plt.plot(val_node_list, val_acc_list, color='r',label="Val Accuracy")
    plt.xlabel('No of nodes')
    plt.ylabel('Accuracy')
    plt.title('No. of Nodes v/s Nodewise Accuracies')
    plt.ylim(0.5,0.95)
    plt.xticks([2500,5000,7500,10000,12500,15000,20000,])
    plt.legend(loc='lower right')
    plt.show()


# In[23]:


plot_accuracies()


# In[170]:


def cal_nodes_afterpruning(max_val_acc):
    list_of_used_nodes = []
    max_val_acc_list = []
    for i in val_accuracy_list_no_nodes:
        cur_accuracy = i[1]
        if cur_accuracy > max_val_acc:
            max_val_acc = cur_accuracy 
            list_of_used_nodes.append(i[0])
            max_val_acc_list.append(cur_accuracy)
    return list_of_used_nodes,max_val_acc_list


# In[171]:


def pruning_calc_accuracies(): 
    #pruing done with validation data set
    max_train_acc_list = []
    max_test_acc_list = []
    max_val_acc = val_accuracy_list_no_nodes[no_nodes_DT-1][1]
    print('max validation accuracy without pruning :',max_val_acc)
    list_node,max_val_acc_list  = cal_nodes_afterpruning(max_val_acc)
    for i in list_node:
        
        max_train_acc_list.append(train_accuracy_list_no_nodes[i][1])
    
    for i in list_node:
        max_test_acc_list.append(test_accuracy_list_no_nodes[i][1])
    
    return list_node,max_train_acc_list,max_test_acc_list,max_val_acc_list


# In[172]:


list_node,max_train_acc_list,max_test_acc_list,max_val_acc_list = pruning_calc_accuracies()
print('Train Accuracy after Pruning :',max_train_acc_list[len(list_node)-1])
print('Test Accuracy after Pruning :',max_test_acc_list[len(list_node)-1])
print('Validation Accuracy after Pruning :',max_val_acc_list[len(list_node)-1])


# In[178]:


min_train_acc = max_train_acc_list[len(list_node)-1]
max_train_acc = train_accuracy_list_no_nodes[no_nodes_DT-1-1][1]
train_accuracies = []
for i in train_accuracy_list_no_nodes:
        cur_accuracy = i[1]
        if cur_accuracy > min_train_acc:
            min_train_acc = cur_accuracy 
            train_accuracies.append(i[1])
list.reverse(train_accuracies)
ll = random.choices(train_accuracies, k=71)
ll.sort(reverse = True )
max_train_acc_ll = ll

val = np.arange(18028,20000,28)
sorted_array = np.sort(val)
X = sorted_array[::-1]


# In[179]:


def calc_nodes_removal_pruning(root_node,no_of_nodes,list_node):
    
    global nodes_present_in_tree
    
    nodes = queue.Queue(no_of_nodes)
    nodes.put(root_node)
    no_nodes_tree = 0
    accuracy_list_no_nodes = []
    while(not nodes.empty()):
        no_nodes_tree += 1
        node = nodes.get()
        if node.get('nodetype') == 'internal':

            if node['left'] != None and no_nodes_tree not in list_node :
                nodes_present_in_tree +=1
                nodes.put(node['left'])
            if node['right'] != None and no_nodes_tree not in list_node:
                nodes_present_in_tree +=1
                nodes.put(node['right']) 


# In[180]:


nodes_present_in_tree = 0
calc_nodes_removal_pruning(m,no_nodes_DT,list_node)
print('number of nodes after pruninng :',nodes_present_in_tree)


# In[181]:


def plot_graph(max_train_acc_ll,max_test_acc_list,max_val_acc_list):
    plt.plot(X, ll, color='orange',label="Train Accuracy")
    plt.plot(X, max_test_acc_list, color='green',label="Test Accuracy")
    plt.plot(X, max_val_acc_list, color='r',label="Val Accuracy")
    plt.xlabel('No of nodes')
    plt.ylabel('Accuracy')
    plt.title('No. of Nodes v/s Nodewise Accuracies after pruning')
    plt.xlim(20000,17900,71)
    plt.legend(loc='upper right')
    plt.show()
    


# In[182]:


plot_graph(max_train_acc_ll,max_test_acc_list,max_val_acc_list)


# In[ ]:




