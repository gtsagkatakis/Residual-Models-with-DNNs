# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 15:05:46 2018

@author: Greg
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Input, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import scipy.io as scio
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from numpy import savetxt
import random as python_random
import tensorflow as tf
import pickle

input_dir='Data_in_mat'
# if len(sys.argv) > 1:
#     input_dir = sys.argv[1]
# else:
#     input_dir = '.'
    
np.random.seed(123)
python_random.seed(123)
# tf.set_random_seed(1234)
tf.random.set_seed(1234)


total_num_epochs=100 #1000
epoch_num=1

def new_metric(y_true, y_pred):
    crng=np.arange(0,50)
    crng2=np.expand_dims(crng,axis=1)
    crng3 = np.repeat(crng2,100,axis=1)
    crng4 = K.transpose(K.constant(crng3 ))
    mn_actual  = K.sum(crng4*y_pred,axis=1)
    y_true_max=K.cast(K.argmax(y_true,axis=1),'float32')
    return K.mean(K.abs(y_true_max-mn_actual))

def mean_JS(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    a2=K.sum([y_true,y_pred], axis=0)
    M=a2*0.5
    M= K.expand_dims(M, 0)
    
    b1=K.sum(y_true * K.log(y_true / M), axis=-1)
    c1=b1*0.5
    
    b2=K.sum(y_pred * K.log(y_pred / M), axis=-1)
    c2=b2*0.5
    
    c1A= K.expand_dims(c1, 0)
    c2A= K.expand_dims(c2, 0)
    
    d1=K.concatenate([c1A,c2A],axis=0)
    d1=K.sum(d1, axis=0)
    
    
    e1 = -K.sum(y_pred * K.log(y_pred)  , axis=1)
    

    d=K.mean(K.sum(0.85*d1+0.15*e1, axis=0))
    
    return d

    

num_of_examples=90000

images=np.zeros([num_of_examples,100,100])

mat = scio.loadmat(input_dir+'/images_May2020_AllModels_30K_dist_part1_analytic_newUnc_newPS.mat')
images[0:30000,:,:]=mat.get('I_part1')
mat = scio.loadmat(input_dir+'/images_May2020_AllModels_30K_dist_part2_merger_newUnc_newPS.mat')
images[30000:60000,:,:]=mat.get('I_part2')
mat = scio.loadmat(input_dir+'/images_May2020_AllModels_30K_dist_part3_spiral_newUnc_newPS.mat')
images[60000:90000,:,:]=mat.get('I_part3')


labels_a=np.zeros([num_of_examples,50])
labels_b=np.zeros([num_of_examples,50])
LBL_a_q=np.zeros([num_of_examples,1])
LBL_b_q=np.zeros([num_of_examples,1])
LBL_points=np.zeros([num_of_examples,2])
LBL_class=np.zeros([num_of_examples,1])


mat = scio.loadmat(input_dir+'/labels_May2020_AllModels_30K_dist_part1_analytic.mat')
labels_a[0:30000,:]=mat.get('LBL_a_dist_BRN')
labels_b[0:30000,:]=mat.get('LBL_b_dist_BRN')
LBL_a_q[0:30000,:]=mat.get('LBL_a_q')
LBL_b_q[0:30000,:]=mat.get('LBL_b_q')
LBL_points[0:30000,:]=mat.get('LBL_points')
LBL_class[0:30000,:]=mat.get('LBL_type')


mat = scio.loadmat(input_dir+'/labels_May2020_AllModels_30K_dist_part2_merger.mat')
labels_a[30000:60000,:]=mat.get('LBL_a_dist_BRN')
labels_b[30000:60000,:]=mat.get('LBL_b_dist_BRN')
LBL_a_q[30000:60000,:]=mat.get('LBL_a_q')
LBL_b_q[30000:60000,:]=mat.get('LBL_b_q')
LBL_points[30000:60000,:]=mat.get('LBL_points')
LBL_class[30000:60000,:]=mat.get('LBL_type')

mat = scio.loadmat(input_dir+'/labels_May2020_AllModels_30K_dist_part3_spiral.mat')
labels_a[60000:90000,:]=mat.get('LBL_a_dist_BRN')
labels_b[60000:90000,:]=mat.get('LBL_b_dist_BRN')
LBL_a_q[60000:90000,:]=mat.get('LBL_a_q')
LBL_b_q[60000:90000,:]=mat.get('LBL_b_q')
LBL_points[60000:90000,:]=mat.get('LBL_points')
LBL_class[60000:90000,:]=mat.get('LBL_type')

[n1,n2,n3]=images.shape


images=images.reshape(n1, n2, n3, 1)

num_train = int(0.8 * n1)   # percentage of data for training
num_test  = int(0.2 * n1)    # percentage of data for testing


# with open(input_dir+'/rand_lst','rb') as filehandle:    
#     rand_lst=pickle.load(filehandle)

rand_lst=np.random.permutation(90000)
num_classes=50

train_data=images[rand_lst[0:num_train],:,:,:]
test_data=images[rand_lst[num_train:num_train+num_test],:,:,:]

train_labels_a=labels_a[rand_lst[0:num_train]]
train_labels_b=labels_b[rand_lst[0:num_train]]
LBL_a_q_train=LBL_a_q[rand_lst[0:num_train]]
LBL_b_q_train=LBL_b_q[rand_lst[0:num_train]]
LBL_a_q_dist_train=to_categorical(LBL_a_q_train-1,num_classes=num_classes)
LBL_b_q_dist_train=to_categorical(LBL_b_q_train-1,num_classes=num_classes)
LBL_class_train=LBL_class[rand_lst[0:num_train]]


test_labels_a=labels_a[rand_lst[num_train:num_train+num_test]]
test_labels_b=labels_b[rand_lst[num_train:num_train+num_test]]
LBL_a_q_test=LBL_a_q[rand_lst[num_train:num_train+num_test]]
LBL_b_q_test=LBL_b_q[rand_lst[num_train:num_train+num_test]]
LBL_a_q_dist_test= to_categorical(LBL_a_q_test-1,num_classes=num_classes)
LBL_b_q_dist_test= to_categorical(LBL_b_q_test-1,num_classes=num_classes)
LBL_class_test=LBL_class[rand_lst[num_train:num_train+num_test]]

del images
    

def res_block(in_data,filter_size):
    conv10  = Conv2D(filter_size, (3, 3), activation='relu', strides=(1), padding='same',kernel_initializer='he_uniform')(in_data)
    bn10 = BatchNormalization()(conv10)
    out_data  = MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same')(bn10)
    return out_data 


inputs = Input(( n2, n3, 1),name='main_input')

pool00=res_block(inputs,64)
pool10=res_block(pool00,64)
pool20=res_block(pool10,64)
pool30=res_block(pool20,64)
pool40=res_block(pool30,64)
pool50=res_block(pool40,64)
pool60=res_block(pool50,64)



fl0 = Flatten(name='fl0')(pool60)
Do0 = Dropout(0.5)(fl0) 

fc0  = Dense(128,activation='linear',name='dense0')(Do0)
bn00 = BatchNormalization()(fc0)
Do1 = Dropout(0.5)(bn00) 
Dn0A = Dense(num_classes,activation='softmax',name='est_a')(Do1)


fc0B  = Dense(128,activation='linear',name='dense0B')(Do0)
bn00B = BatchNormalization()(fc0B)
Do1B = Dropout(0.5)(bn00B) 
Dn0B = Dense(num_classes,activation='softmax',name='est_b')(Do1B)

model_1 = Model(inputs=[inputs], outputs=[Dn0A,Dn0B])
optzr =  Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=0)

model_1.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer=optzr, metrics =[new_metric])
# model_1.compile(loss=mean_JS,optimizer=optzr, metrics =[new_metric])


filepath="T_JJS_loss_3Models_weights-improvement_CrossEntropy_AB-{epoch:03d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', period=20)
callbacks_list = [checkpoint]

history=model_1.fit(train_data,[train_labels_a,train_labels_b], batch_size=100, epochs=total_num_epochs,validation_data=[train_data,[LBL_a_q_dist_train,LBL_b_q_dist_train]],callbacks=callbacks_list,shuffle=True)



with open('history_cross_entropy.dump','wb') as filehandle:
    pickle.dump(history.history,filehandle)



crng=np.arange(0,50)

evaluate_on_training=False

if evaluate_on_training==True:
    er_a=np.zeros([num_train,3])
    lbl_mon_a=np.zeros([num_train,4])
    er_b=np.zeros([num_train,3])
    lbl_mon_b=np.zeros([num_train,4])
    num_smp_to_show=num_train
else:
    er_a=np.zeros([num_test,3])
    lbl_mon_a=np.zeros([num_test,4])
    er_b=np.zeros([num_test,3])
    lbl_mon_b=np.zeros([num_test,4])
    num_smp_to_show=num_test



for tt in range(0,num_smp_to_show): 
    if evaluate_on_training==True:
        pred_dist_a_b=model_1.predict(train_data[tt:tt+1,:,:,:])
        pred_dist_a=pred_dist_a_b[0]
        lbl_dist_a=train_labels_a[tt,:]
        lbl_actual_a=LBL_a_q_train[tt]
        pred_dist_b=pred_dist_a_b[1]
        lbl_dist_b=train_labels_b[tt,:]
        lbl_actual_b=LBL_b_q_train[tt]
        lbl_class=LBL_class_train[tt]
        
        
    else:
        pred_dist_a_b=model_1.predict(test_data[tt:tt+1,:,:,:])    
        pred_dist_a=pred_dist_a_b[0]
        lbl_dist_a=test_labels_a[tt,:]
        lbl_actual_a=LBL_a_q_test[tt]
        pred_dist_b=pred_dist_a_b[1]
        lbl_dist_b=test_labels_b[tt,:]
        lbl_actual_b=LBL_b_q_test[tt]
        lbl_class=LBL_class_test[tt]
        
        

        
    mn_pred_a=np.sum(crng*pred_dist_a)
    mn_actual_a=np.sum(crng*lbl_dist_a)
    mn_pred_b=np.sum(crng*pred_dist_b)
    mn_actual_b=np.sum(crng*lbl_dist_b)

    
    er_a[tt,0]=np.abs(mn_pred_a-lbl_actual_a)
    er_a[tt,1]=np.abs(mn_actual_a-lbl_actual_a)
    er_a[tt,2]=lbl_actual_a
    
    er_b[tt,0]=np.abs(mn_pred_b-lbl_actual_b)
    er_b[tt,1]=np.abs(mn_actual_b-lbl_actual_b)
    er_b[tt,2]=lbl_actual_b

    
    lbl_mon_a[tt,0]=lbl_actual_a
    lbl_mon_a[tt,1]=mn_pred_a
    lbl_mon_a[tt,2]=mn_actual_a
    lbl_mon_a[tt,3]=lbl_class
    
    lbl_mon_b[tt,0]=lbl_actual_b
    lbl_mon_b[tt,1]=mn_pred_b
    lbl_mon_b[tt,2]=mn_actual_b
    lbl_mon_b[tt,3]=lbl_class

   
       

print('Error a w/ pred  ',np.mean(er_a[:,0])) # predictions
print('Error a w/ actual', np.mean(er_a[:,1])) # actual labels

print('Error b w/ pred  ',np.mean(er_b[:,0])) # predictions
print('Error b w/ actual', np.mean(er_b[:,1])) # actual labels

print('Error Median a w/ pred  ',np.median(er_a[:,0])) # predictions
print('Error Median a w/ actual', np.median(er_a[:,1])) # actual labels

print('Error Median b w/ pred  ',np.median(er_b[:,0])) # predictions
print('Error Median b w/ actual', np.median(er_b[:,1])) # actual labels

savetxt('errors_a_CE.csv', er_a, delimiter=',')
savetxt('lbl_mon_a_CE.csv', lbl_mon_a, delimiter=',')

savetxt('errors_b_CE.csv', er_b, delimiter=',')
savetxt('lbl_mon_b_CE.csv', lbl_mon_b, delimiter=',')

