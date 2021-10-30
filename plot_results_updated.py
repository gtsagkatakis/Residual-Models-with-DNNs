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
import scipy as scipy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import to_categorical
from numpy import savetxt
import random as python_random
import tensorflow as tf
import pickle
from time import time
# from tensorflow.python.keras.callbacks import TensorBoard
# from datetime import datetime

# logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=logdir)

input_dir='Data_in_mat'

# if len(sys.argv) > 1:
#     input_dir = sys.argv[1]
# else:
#     input_dir = '.'
    
np.random.seed(123)
python_random.seed(123)
# tf.set_random_seed(1234)


total_num_epochs=1
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
# rand_lst=range(0,num_of_examples)
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


# with open('runC0\history.dump','rb') as filehandle:    
#     history_C0=pickle.load(filehandle)
# with open('runC3\history_new_loss_09_01.dump','rb') as filehandle:    
#     history_C3=pickle.load(filehandle)
# #with open('runC4\history_new_loss_85_15.dump','rb') as filehandle:    
# #    history_C4=pickle.load(filehandle)
# with open('runC5\history_new_loss_95_05.dump','rb') as filehandle:    
#     history_C5=pickle.load(filehandle)
# with open('runC6\history_new_loss_09_01.dump','rb') as filehandle:    
#     history_C6=pickle.load(filehandle)
# with open('runD_history_cross_entropy.dump','rb') as filehandle:    
#     history_C7=pickle.load(filehandle)
   

# # A=history_C0.get('val_est_a_new_metric')[0:1000]
# # Af=np.median(strided_app(data, window_len,3),axis=1)

# plt.plot(history_C0.get('val_est_a_new_metric')[0:1000],label='Jensen–Shannon loss',zorder=3)
# plt.plot(history_C3.get('val_est_a_new_metric')[0:1000],label='Jensen–Shannon & Entropy loss',zorder=4)
# plt.plot(history_C7.get('val_est_a_new_metric')[0:1000],label='Cross Entropy',zorder=2)
# #plt.plot(history_C4.get('val_est_a_new_metric'),label='85-15')
# # plt.plot(history_C5.get('val_est_a_new_metric')[0:1000],label='95-5')
# # plt.plot(history_C6.get('val_est_a_new_metric')[0:1000],label='adaptive')
# plt.yscale('linear')
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Prediction disparity')
# plt.show()

# A=np.zeros([3,1000])
# A[0,:]=history_C0.get('val_est_a_new_metric')[0:1000]
# A[1,:]=history_C3.get('val_est_a_new_metric')[0:1000]
# A[2,:]=history_C7.get('val_est_a_new_metric')[0:1000]


# plt.plot(history_C0.get('val_est_b_new_metric')[0:1000],label='Jensen–Shannon loss',zorder=3)
# plt.plot(history_C3.get('val_est_b_new_metric')[0:1000],label='Jensen–Shannon & Entropy loss',zorder=4)
# plt.plot(history_C7.get('val_est_b_new_metric')[0:1000],label='Cross Entropy',zorder=2)
# #plt.plot(history_C4.get('val_est_a_new_metric'),label='85-15')
# # plt.plot(history_C5.get('val_est_a_new_metric')[0:1000],label='95-5')
# # plt.plot(history_C6.get('val_est_a_new_metric')[0:1000],label='adaptive')
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Prediction disparity')
# plt.show()

# B=np.zeros([3,1000])
# B[0,:]=history_C0.get('val_est_b_new_metric')[0:1000]
# B[1,:]=history_C3.get('val_est_b_new_metric')[0:1000]
# B[2,:]=history_C7.get('val_est_b_new_metric')[0:1000]

# np.save('A.npy',A)
# np.save('B.npy',B)

# print(np.mean(history_C0.get('val_est_a_new_metric')[900:1000]))
# print(np.mean(history_C3.get('val_est_a_new_metric')[900:1000]))
# print(np.mean(history_C5.get('val_est_a_new_metric')[900:1000]))
# print(np.mean(history_C6.get('val_est_a_new_metric')[900:1000]))

# print(np.median(history_C0.get('val_est_a_new_metric')[900:1000]))
# print(np.median(history_C3.get('val_est_a_new_metric')[900:1000]))
# print(np.median(history_C5.get('val_est_a_new_metric')[900:1000]))
# print(np.median(history_C6.get('val_est_a_new_metric')[900:1000]))


# plt.plot(history_C0.get('val_est_b_new_metric')[00:1000],label='single term',zorder=2)
# plt.plot(history_C3.get('val_est_b_new_metric')[00:1000],label='90-10')
# #plt.plot(history_C4.get('val_est_a_new_metric'),label='85-15')
# plt.plot(history_C5.get('val_est_b_new_metric')[00:1000],label='95-5')
# plt.plot(history_C6.get('val_est_b_new_metric')[00:1000],label='adaptive',zorder=3)
# plt.legend()

# print(np.mean(history_C0.get('val_est_b_new_metric')[900:1000]))
# print(np.mean(history_C3.get('val_est_b_new_metric')[900:1000]))
# print(np.mean(history_C5.get('val_est_b_new_metric')[900:1000]))
# print(np.mean(history_C6.get('val_est_b_new_metric')[900:1000]))

# print(np.median(history_C0.get('val_est_b_new_metric')[900:1000]))
# print(np.median(history_C3.get('val_est_b_new_metric')[900:1000]))
# print(np.median(history_C5.get('val_est_b_new_metric')[900:1000]))
# print(np.median(history_C6.get('val_est_b_new_metric')[900:1000]))



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

model_1.compile(loss=mean_JS,optimizer=optzr, metrics =[new_metric])


#filepath="T_JJS_loss_3Models_weights-improvement_NewLoss_09_01_AB-{epoch:03d}-{val_loss:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', period=20)
#callbacks_list = [checkpoint]

#history=model_1.fit(train_data,[train_labels_a,train_labels_b], batch_size=100, epochs=total_num_epochs,validation_data=[train_data,[LBL_a_q_dist_train,LBL_b_q_dist_train]],callbacks=callbacks_list,shuffle=True)

# model_1.load_weights("runC3\T_JJS_loss_3Models_weights-improvement_NewLoss_09_01_AB-1000-1.57.hdf5")
model_1.load_weights("T_JJS_loss_3Models_weights-improvement_NewLoss_09_01_AB-1000-1.15.hdf5")



crng=np.arange(0,50)

evaluate_on_training=False

if evaluate_on_training==True:
    er_a=np.zeros([num_train,3])
    lbl_mon_a=np.zeros([num_train,4])
    er_b=np.zeros([num_train,3])
    lbl_mon_b=np.zeros([num_train,4])
    num_smp_to_show=num_train
    uncertainty_target=np.zeros([num_train,1])
    uncertainty_pred=np.zeros([num_train,1])
    uncertainty_ratio=np.zeros([num_train,1])
    pred_dist_a=np.zeros([num_train,50])
    pred_dist_b=np.zeros([num_train,50])
    
    lbl_dist_a=np.zeros([num_train,50])
    lbl_dist_b=np.zeros([num_train,50])
    
    
else:
    er_a=np.zeros([num_test,3])
    lbl_mon_a=np.zeros([num_test,4])
    er_b=np.zeros([num_test,3])
    lbl_mon_b=np.zeros([num_test,4])
    num_smp_to_show=num_test
    uncertainty_target=np.zeros([num_test,1])
    uncertainty_pred=np.zeros([num_test,1])
    uncertainty_ratio=np.zeros([num_test,1])
    pred_dist_a=np.zeros([num_test,50])
    pred_dist_b=np.zeros([num_test,50])

    lbl_dist_a=np.zeros([num_test,50])
    lbl_dist_b=np.zeros([num_test,50])



if evaluate_on_training==True:
        pred_dist_a_b_list=model_1.predict(train_data)
else:
        pred_dist_a_b_list=model_1.predict(test_data)    



for tt in range(0,num_smp_to_show): 
    
    if np.mod(tt,10)==0:
        print(tt)    
        
    if evaluate_on_training==True:
        # pred_dist_a_b=model_1.predict(train_data[tt:tt+1,:,:,:])
        # pred_dist_a_b=pred_dist_a_b_list[0][:,tt]
        pred_dist_a=pred_dist_a_b_list[0][tt,:]
        lbl_dist_a=train_labels_a[tt,:]
        lbl_actual_a=LBL_a_q_train[tt]
        pred_dist_b=pred_dist_a_b_list[1][tt,:]
        lbl_dist_b=train_labels_b[tt,:]
        lbl_actual_b=LBL_b_q_train[tt]
        lbl_class=LBL_class_train[tt]
        
        
    else:
        # pred_dist_a_b=model_1.predict(test_data[tt:tt+1,:,:,:])    
        # pred_dist_a_b=pred_dist_a_b_list[0:1][tt]    
        pred_dist_a=pred_dist_a_b_list[0][tt,:]
        lbl_dist_a=test_labels_a[tt,:]
        lbl_actual_a=LBL_a_q_test[tt]
        pred_dist_b=pred_dist_a_b_list[1][tt,:]
        lbl_dist_b=test_labels_b[tt,:]
        lbl_actual_b=LBL_b_q_test[tt]
        lbl_class=LBL_class_test[tt]
        
        

       
     
    mn_pred_a=np.sum(crng*pred_dist_a)
    mn_target_a=np.sum(crng*lbl_dist_a)
    mn_pred_b=np.sum(crng*pred_dist_b)
    mn_target_b=np.sum(crng*lbl_dist_b)

    
    er_a[tt,0]=np.abs(mn_pred_a-lbl_actual_a)
    er_a[tt,1]=np.abs(mn_target_a-lbl_actual_a)
    er_a[tt,2]=lbl_actual_a
    
    er_b[tt,0]=np.abs(mn_pred_b-lbl_actual_b)
    er_b[tt,1]=np.abs(mn_target_b-lbl_actual_b)
    er_b[tt,2]=lbl_actual_b
    
    # area_target=np.outer(lbl_dist_a,lbl_dist_b)
    # plt.imshow(area_target)
    # uncertainty_target[tt]=len(np.where(area_target>0)[0])/len(np.where(area_target>=0)[0])
    # plt.colorbar()
    # plt.show()
    
    area_predict=np.outer(pred_dist_a,pred_dist_b)
    aa = np.linspace(-5.0,-2.0,50)
    bb = np.linspace(-8.0,-3.0,50)
    x,y = np.meshgrid(bb,aa)
    
    # sigmas  = [0.997,0.95,0.68]
    # clevels = [0,0,0]
    sigmas  = [0.997,0.95]
    clevels = [0,0]

    maxa = np.amax(area_predict)
    thres = 0.01*maxa
    dthres = 0.01*maxa
    ind_sigma = 0
    # while ind_sigma < 3 and thres<maxa:
    while ind_sigma < 2 and thres<maxa:
        # masked = np.ma.masked_where(area_predict<thres,area_predict)
        # masked_int=masked.mask.astype(int)
        # mysum = np.sum(masked_int)
        # print(mysum)
        masked = np.ma.masked_where(area_predict<thres,area_predict)
        mysum = np.sum(masked)
        if mysum < sigmas[ind_sigma]:
            clevels[ind_sigma] = thres - dthres
            ind_sigma = ind_sigma + 1
        thres = thres + dthres

    uncertainty_pred[tt]=len(np.where(area_predict>clevels[1])[0])/len(np.where(area_predict>=0)[0])
    # uncertainty_pred[tt]=len(np.where(area_predict>clevels[2])[0])/len(np.where(area_predict>=0)[0])

    area_target=np.outer(lbl_dist_a,lbl_dist_b)
    aa = np.linspace(-5.0,-2.0,50)
    bb = np.linspace(-8.0,-3.0,50)
    x,y = np.meshgrid(bb,aa)
    
    # sigmas  = [0.997,0.95,0.68]
    # clevels = [0,0,0]
    sigmas  = [0.997,0.95]
    clevels = [0,0]

    maxa = np.amax(area_target)
    thres = 0.01*maxa
    dthres = 0.01*maxa
    ind_sigma = 0
    # while ind_sigma < 3 and thres<maxa:
    while ind_sigma < 2 and thres<maxa:
        masked = np.ma.masked_where(area_target<thres,area_target)
        mysum = np.sum(masked)
        # print(mysum)
        if mysum < sigmas[ind_sigma]:
            clevels[ind_sigma] = thres - dthres
            ind_sigma = ind_sigma + 1
        thres = thres + dthres

    # uncertainty_target[tt]=len(np.where(area_target>clevels[2])[0])/len(np.where(area_target>=0)[0])
    uncertainty_target[tt]=len(np.where(area_target>clevels[1])[0])/len(np.where(area_target>=0)[0])
    uncertainty_ratio[tt]=uncertainty_pred[tt]/uncertainty_target[tt]


# plt.plot(uncertainty_target,label='target')
# plt.plot(uncertainty_pred,label='pred')
# plt.legend()
# plt.show()

    # lbl_mon_a[tt,0]=lbl_actual_a
    # lbl_mon_a[tt,1]=mn_pred_a
    # lbl_mon_a[tt,2]=mn_target_a
    # lbl_mon_a[tt,3]=lbl_class
    
    # lbl_mon_b[tt,0]=lbl_actual_b
    # lbl_mon_b[tt,1]=mn_pred_b
    # lbl_mon_b[tt,2]=mn_target_b
    # lbl_mon_b[tt,3]=lbl_class

    
            
    if tt<0:
        print("A: Case %d, Error w/ pred %.1f, Error w/ true %.1f" %(tt, er_a[tt,0], er_a[tt,1]))
        plt.subplot(2,1,1)
        plt.plot(pred_dist_a,'b--')
        plt.plot(lbl_dist_a,'rx-')
        plt.plot(mn_target_a,0.1,'kd')
        plt.plot(mn_pred_a,0.1,'gd')
        plt.plot(lbl_actual_a,0.1,'yp')
        plt.legend(['Prediction','GT distribution','Mean value','Pred value','GT value'])
        plt.title("A: Case %d, Class %d, Error pred %.1f, Error true %.1f" %(tt,lbl_mon_a[tt,3], er_a[tt,0], er_a[tt,1]))
        

        
        plt.subplot(2,1,2)
        tmp=np.cumsum(pred_dist_a)
        plt.plot(tmp,'b--')
        tmp2=np.cumsum(lbl_dist_a)
        plt.plot(tmp2,'r-')
        plt.show()
        
        mn_pred_a=np.argmin(np.abs(tmp-0.15))
        mx_pred_a=np.argmin(np.abs(tmp-0.85))
        err_var=mx_pred_a-mn_pred_a
        
        # mn_gt_a=np.argmin(np.abs(tmp2-0.05))
        # mx_gt_a=np.argmin(np.abs(tmp2-0.95))
        mn_gt_a=np.min(np.where(lbl_dist_a>0))
        mx_gt_a=np.max(np.where(lbl_dist_a>0))

        err_var2=mx_gt_a-mn_gt_a
        
        tmp3=np.cumsum(pred_dist_b)
        # plt.plot(tmp,'b--')
        tmp4=np.cumsum(lbl_dist_b)
        # plt.plot(tmp2,'r-')
        
        mn_pred_b=np.argmin(np.abs(tmp3-0.15))
        mx_pred_b=np.argmin(np.abs(tmp3-0.85))
        err_var=mx_pred_b-mn_pred_b
        
        # mn_gt_b=np.argmin(np.abs(tmp4-0.05))
        # mx_gt_b=np.argmin(np.abs(tmp4-0.95))
        mn_gt_b=np.min(np.where(lbl_dist_b>0))
        mx_gt_b=np.max(np.where(lbl_dist_b>0))
        err_var2=mx_gt_b-mn_gt_b
        
        
        # plt.title("A: err pred: %d, err gt: %d" %(err_var,err_var2))
        
        # plt.subplots_adjust(hspace=0.5)
        # plt.show()
    
        # fig,ax=plt.subplots()
        # a=np.outer(pred_dist_a[0,:],pred_dist_b[0,:])
        # b=np.outer(lbl_dist_a,lbl_dist_b)
        # # plt.imshow(b, alpha=0.4)
        # plt.imshow(a)
        # rect=plt.Rectangle((mn_gt_b,mn_gt_a), mx_gt_b-mn_gt_b, mx_gt_a-mn_gt_a,fill=False,edgecolor='r' )
        # ax.add_patch(rect)
        # plt.scatter(mn_actual_b,mn_actual_a,color='r',marker='x')
        # plt.show()
        
 
#
#plt.plot(er_a[:,0],'r')
#plt.plot(er_a[:,1],'b')
#plt.legend(['Error for a: w/ pred','Error w/ actual'])
#plt.show()
#
#plt.plot(er_b[:,0],'r')
#plt.plot(er_b[:,1],'b')
#plt.legend(['Error for b: w/ pred','Error w/ actual'])
#plt.show()

print('Error a w/ pred  ',np.mean(er_a[:,0])) # predictions
print('Error a w/ actual', np.mean(er_a[:,1])) # actual labels

print('Error b w/ pred  ',np.mean(er_b[:,0])) # predictions
print('Error b w/ actual', np.mean(er_b[:,1])) # actual labels

print('Error Median a w/ pred  ',np.median(er_a[:,0])) # predictions
print('Error Median a w/ actual', np.median(er_a[:,1])) # actual labels

print('Error Median b w/ pred  ',np.median(er_b[:,0])) # predictions
print('Error Median b w/ actual', np.median(er_b[:,1])) # actual labels

combined_err_prediction=np.zeros([num_smp_to_show,1])
combined_err_target=np.zeros([num_smp_to_show,1])
for tt in range(0,num_smp_to_show):
    combined_err_prediction[tt,:]=np.sqrt(np.square(er_a[tt,0])+np.square(er_b[tt,0]))
    combined_err_target[tt,:]=np.sqrt(np.square(er_a[tt,1])+np.square(er_b[tt,1]))

map_target_unc=np.zeros([50,50])
map_target_dst=np.zeros([50,50])
map_pred_unc=np.zeros([50,50])
map_pred_dst=np.zeros([50,50])


idx_x=0
idx_y=0
# bins_a=np.histogram(er_a[:,2],5)[1]
# bins_b=np.histogram(er_b[:,2],5)[1]
for qq_a in range(1,51,1):
    for qq_b in range(1,51,1):
        # range_a=np.where(er_a[:,2]>=qq_a) # and np.where(er_a[:,2]<=qq_a)
        # range_b=np.where(er_b[:,2]>=qq_b) # and np.where(er_b[:,2]<=qq_b)
        # a_pred=uncertainty_pred[np.intersect1d(range_a,range_b)]
        # tmp_map_pred[idx_x,idx_y]=np.mean(a_pred)
        
        # a_target=uncertainty_target[np.intersect1d(range_a,range_b)]
        # tmp_map_target[idx_x,idx_y]=np.mean(a_target)


        # range_a = np.union1d(np.where(er_a[:,2]==qq_a),np.where(er_a[:,2]==qq_a+1))
        # range_b = np.union1d(np.where(er_b[:,2]==qq_b),np.where(er_b[:,2]==qq_b+1))
        
        range_a = np.where(er_a[:,2]==qq_a)
        range_b = np.where(er_b[:,2]==qq_b)


        a_target=uncertainty_target[np.intersect1d(range_a ,range_b )]
        map_target_unc[idx_x,idx_y]=np.mean(a_target)
        
        a_target=combined_err_target[np.intersect1d(range_a ,range_b )]
        map_target_dst[idx_x,idx_y]=np.mean(a_target)
        
 
     
        a_pred=uncertainty_pred[np.intersect1d(range_a,range_b)]   
        map_pred_unc[idx_x,idx_y]=np.mean(a_pred)
        
        a_pred=combined_err_prediction[np.intersect1d(range_a,range_b)]   
        map_pred_dst[idx_x,idx_y]=np.mean(a_pred)
        


        # a_pred=combined_err_prediction[np.intersect1d(np.where(er_a[:,2]==qq_a),np.where(er_b[:,2]==qq_b))]
        # tmp_map_pred[idx_x,idx_y]=np.mean(a_pred)
        
        # a_target=combined_err_target[np.intersect1d(np.where(er_a[:,2]==qq_a),np.where(er_b[:,2]==qq_b))]
        # tmp_map_target[idx_x,idx_y]=np.mean(a_target)



        # a=np.intersect1d(np.where(er_a[:,2]>=qq_a), np.where(er_a[:,2]<qq_a+10))
        # b=np.intersect1d(np.where(er_b[:,2]>=qq_b), np.where(er_b[:,2]<qq_b+10))
        # c=np.intersect1d(a,b)
        # # tmp_map[idx_x,idx_y]=np.mean(uncertainty_target[c])
        # tmp_map[idx_x,idx_y]=np.mean(uncertainty_pred[c])
        
        
        idx_y+=1
        
    idx_x+=1
    idx_y=0


if evaluate_on_training==True:
    np.save('Distance_training_prediction_2sigma',map_pred_dst)
    np.save('Distance_training_target_2sigma',map_target_dst)

    np.save('Uncertainty_training_prediction_2sigma', map_pred_unc)
    np.save('Uncertainty_training_target_2sigma', map_target_unc)

else:
    np.save('Distance_validation_prediction_2sigma',map_pred_dst)
    np.save('Distance_validation_target_2sigma',map_target_dst)

    np.save('Uncertainty_validation_prediction_2sigma', map_pred_unc)
    np.save('Uncertainty_validation_target_2sigma', map_target_unc)



sys.exit()

# grid_z2 = griddata(tmp_map_pred, tmp_map_pred, (grid_x, grid_y), method='cubic')


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=5, random_state=0)

imp.fit(map_pred_unc)
map_pred_unc_rec=imp.transform(map_pred_unc)

imp.fit(map_target_unc)
map_target_unc_rec=imp.transform(map_target_unc)


plt.imshow(np.flipud(map_pred_unc_rec))
# plt.imshow(np.flipud(tmp_map_pred), vmin=0, vmax=0.25,cmap='rainbow')
# plt.imshow(np.flipud(tmp_map_pred), vmin=0, vmax=10,cmap='rainbow')
plt.xlabel('A')
plt.ylabel('b')
# plt.xticks([0,10,20,30,40,50],['-8','-7','-6','-5','-4','-3'])
# plt.yticks([0,10,20,30,40,50],['-5','-4.4','-3.8','-3.2','-2.6','-2'])
# plt.title('Distance error for prediction')
# plt.title('Target uncertainty')
plt.title('Prediction uncertainty')
plt.colorbar()
plt.show()


plt.imshow(np.flipud(map_target_unc_rec))
# plt.imshow(np.flipud(tmp_map_target), vmin=0, vmax=0.25,cmap='rainbow')
# plt.imshow(np.flipud(tmp_map_target), vmin=0, vmax=10,cmap='rainbow')
plt.xlabel('A')
plt.ylabel('b')
# plt.xticks([0,10,20,30,40,50],['-8','-7','-6','-5','-4','-3'])
# plt.yticks([0,10,20,30,40,50],['-5','-4.4','-3.8','-3.2','-2.6','-2'])
# plt.title('Distance error for target')
plt.title('Target uncertainty')
# plt.title('Prediction uncertainty')
plt.colorbar()
plt.show()

# np.save('map_pred_unc.npy',map_pred_unc)
# np.save('map_target_unc.npy',map_target_unc)
# np.save('map_target_dst.npy',map_target_dst)
# np.save('map_pred_dst.npy',map_pred_dst)




sys.exit()

# plt.figure()
# plt.hist2d(er_a[:,0],er_b[:,0], bins=5)
# plt.show()

# a=np.argsort(er_a[:,0])
# for qq in range(num_smp_to_show-20,num_smp_to_show):
#     plt.title("Error %f, Class %d" %( er_a[a[qq],0],lbl_mon_a[a[qq],3] ))
#     b=test_data[a[qq],:,:,0]
#     plt.imshow(b)
#     plt.show()
    
    

# for qq in range(0,20):
#     plt.title("Error %f, Class %d" %( er_a[a[qq],0],lbl_mon_a[a[qq],3] ))
#     b=test_data[a[qq],:,:,0]
#     plt.imshow(b)
#     plt.show()


map_pred_dst=np.load('Uncertainty_training_prediction.npy')
map_target_dst=np.load('Uncertainty_training_target.npy')


plt.imshow(np.flipud(map_pred_dst),vmin=0, vmax=0.25)
plt.xlabel('A')
plt.ylabel('b')
plt.xticks([0,10,20,30,40,50],['-8','-7','-6','-5','-4','-3'])
plt.yticks([0,10,20,30,40,50],['-5','-4.4','-3.8','-3.2','-2.6','-2'])
plt.title('Uncertainty for prediction')
plt.colorbar()
plt.show()


plt.imshow(np.flipud(map_target_dst),vmin=0, vmax=0.25)
plt.xlabel('A')
plt.ylabel('b')
plt.xticks([0,10,20,30,40,50],['-8','-7','-6','-5','-4','-3'])
plt.yticks([0,10,20,30,40,50],['-5','-4.4','-3.8','-3.2','-2.6','-2'])
plt.title('Uncertainty for target')
plt.colorbar()
plt.show()