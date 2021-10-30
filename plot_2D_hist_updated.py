import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense, Input, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import scipy.io as scio
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



# if len(sys.argv) > 1:
#     input_dir = sys.argv[1]
# else:
#     input_dir = '.'

input_dir='Data_in_mat'
    
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
labels_a[0:30000,:]=mat.get('LBL_a_dist_BRN_F')
labels_b[0:30000,:]=mat.get('LBL_b_dist_BRN_F')
LBL_a_q[0:30000,:]=mat.get('LBL_a_q_F')
LBL_b_q[0:30000,:]=mat.get('LBL_b_q_F')
LBL_points[0:30000,:]=mat.get('LBL_points_F')
LBL_class[0:30000,:]=mat.get('LBL_type_F')


mat = scio.loadmat(input_dir+'/labels_May2020_AllModels_30K_dist_part2_merger.mat')
labels_a[30000:60000,:]=mat.get('LBL_a_dist_BRN_F')
labels_b[30000:60000,:]=mat.get('LBL_b_dist_BRN_F')
LBL_a_q[30000:60000,:]=mat.get('LBL_a_q_F')
LBL_b_q[30000:60000,:]=mat.get('LBL_b_q_F')
LBL_points[30000:60000,:]=mat.get('LBL_points_F')
LBL_class[30000:60000,:]=mat.get('LBL_type_F')

mat = scio.loadmat(input_dir+'/labels_May2020_AllModels_30K_dist_part3_spiral.mat')
labels_a[60000:90000,:]=mat.get('LBL_a_dist_BRN_F')
labels_b[60000:90000,:]=mat.get('LBL_b_dist_BRN_F')
LBL_a_q[60000:90000,:]=mat.get('LBL_a_q_F')
LBL_b_q[60000:90000,:]=mat.get('LBL_b_q_F')
LBL_points[60000:90000,:]=mat.get('LBL_points_F')
LBL_class[60000:90000,:]=mat.get('LBL_type_F')

[n1,n2,n3]=images.shape


images=images.reshape(n1, n2, n3, 1)

num_train = int(0.8 * n1)   # percentage of data for training
num_test  = int(0.2 * n1)    # percentage of data for testing


# with open(input_dir+'/rand_lst','rb') as filehandle:    
#     rand_lst=pickle.load(filehandle)
# rand_lst=range(0,num_of_examples)
rand_lst=np.random.permutation(90000)

for tt in range(0,10):
    plt.imshow(np.squeeze(images[rand_lst[tt],:,:])) 
    plt.axis('off')
    plt.show()
    


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


with open('runC0_history.dump','rb') as filehandle:    
    history_C0=pickle.load(filehandle)
with open('runC3_history_new_loss_09_01.dump','rb') as filehandle:    
    history_C3=pickle.load(filehandle)
#with open('runC4\history_new_loss_85_15.dump','rb') as filehandle:    
#    history_C4=pickle.load(filehandle)
with open('runC5_history_new_loss_95_05.dump','rb') as filehandle:    
    history_C5=pickle.load(filehandle)
with open('runC6_history_new_loss_09_01.dump','rb') as filehandle:    
    history_C6=pickle.load(filehandle)
   


plt.plot(history_C0.get('val_est_a_new_metric')[900:1000],label='single term',zorder=3)
plt.plot(history_C3.get('val_est_a_new_metric')[900:1000],label='90-10')
#plt.plot(history_C4.get('val_est_a_new_metric'),label='85-15')
plt.plot(history_C5.get('val_est_a_new_metric')[900:1000],label='95-5')
plt.plot(history_C6.get('val_est_a_new_metric')[900:1000],label='adaptive')
plt.legend()

print(np.mean(history_C0.get('val_est_a_new_metric')[900:1000]))
print(np.mean(history_C3.get('val_est_a_new_metric')[900:1000]))
print(np.mean(history_C5.get('val_est_a_new_metric')[900:1000]))
print(np.mean(history_C6.get('val_est_a_new_metric')[900:1000]))

print(np.median(history_C0.get('val_est_a_new_metric')[900:1000]))
print(np.median(history_C3.get('val_est_a_new_metric')[900:1000]))
print(np.median(history_C5.get('val_est_a_new_metric')[900:1000]))
print(np.median(history_C6.get('val_est_a_new_metric')[900:1000]))


plt.plot(history_C0.get('val_est_b_new_metric')[00:1000],label='single term',zorder=2)
plt.plot(history_C3.get('val_est_b_new_metric')[00:1000],label='90-10')
#plt.plot(history_C4.get('val_est_a_new_metric'),label='85-15')
plt.plot(history_C5.get('val_est_b_new_metric')[00:1000],label='95-5')
plt.plot(history_C6.get('val_est_b_new_metric')[00:1000],label='adaptive',zorder=3)
plt.legend()

print(np.mean(history_C0.get('val_est_b_new_metric')[900:1000]))
print(np.mean(history_C3.get('val_est_b_new_metric')[900:1000]))
print(np.mean(history_C5.get('val_est_b_new_metric')[900:1000]))
print(np.mean(history_C6.get('val_est_b_new_metric')[900:1000]))

print(np.median(history_C0.get('val_est_b_new_metric')[950:1000]))
print(np.median(history_C3.get('val_est_b_new_metric')[950:1000]))
print(np.median(history_C5.get('val_est_b_new_metric')[950:1000]))
print(np.median(history_C6.get('val_est_b_new_metric')[950:1000]))



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

# model_1.load_weights("runC3\T_JJS_loss_3Models_weights-improvement_NewLoss_09_01_AB-960-1.57.hdf5")
model_1.load_weights("T_JJS_loss_3Models_weights-improvement_NewLoss_09_01_AB-1000-1.15.hdf5")




crng=np.arange(0,50)

# tt=325
tt=0
tt=30

# for tt in range(0,100):
for tt in [32]:

    pred_dist_a_b=model_1.predict(test_data[tt:tt+1,:,:,:])    
    
    pred_dist_a=pred_dist_a_b[0]
    lbl_dist_a=test_labels_a[tt,:]
    lbl_actual_a=LBL_a_q_test[tt]
    pred_dist_b=pred_dist_a_b[1]
    lbl_dist_b=test_labels_b[tt,:]
    lbl_actual_b=LBL_b_q_test[tt]
    lbl_class=LBL_class_test[tt]
    
    mn_actual_a=np.sum(crng*lbl_dist_a)
    mn_actual_b=np.sum(crng*lbl_dist_b)
    mn_pred_a=np.sum(crng*pred_dist_a[0,:])
    mn_pred_b=np.sum(crng*pred_dist_b[0,:])
    
    
    # plt.subplot(2,1,1)
    # plt.plot(pred_dist_a[0,:],'b--')
    # plt.plot(lbl_dist_a,'rx-')
    # plt.plot(mn_actual_a,0.1,'kd')
    # plt.plot(mn_pred_a,0.1,'gd')
    # plt.plot(lbl_actual_a,0.1,'yp')
    # plt.legend(['Prediction','GT distribution','Mean value','Pred value','GT value'])
       
    
    
    # plt.subplot(2,1,2)
    # plt.plot(pred_dist_b[0,:],'b--')
    # plt.plot(lbl_dist_b,'rx-')
    # plt.plot(mn_actual_b,0.1,'kd')
    # plt.plot(mn_pred_b,0.1,'gd')
    # plt.plot(lbl_actual_b,0.1,'yp')
    # plt.legend(['Prediction','GT distribution','Mean value','Pred value','GT value'])
    # plt.show()  
    
    # np.save('pred_dist_a.npy',pred_dist_a)  
    # np.save('lbl_dist_a.npy',lbl_dist_a)  
    # np.save('lbl_actual_a.npy',lbl_actual_a)  
    
    # np.save('pred_dist_b.npy',pred_dist_b)  
    # np.save('lbl_dist_b.npy',lbl_dist_b)  
    # np.save('lbl_actual_b.npy',lbl_actual_b)  
    
    # np.save('mn_pred_a.npy',mn_pred_a)  
    # np.save('mn_actual_a.npy',mn_actual_a)  
    # np.save('mn_pred_b.npy',mn_pred_b)  
    # np.save('mn_actual_b.npy',mn_actual_b)  
    
    
    
    
    # pred_dist_a=np.load('pred_dist_a.npy')
    # lbl_dist_a=np.load('lbl_dist_a.npy')
    # lbl_actual_a=np.load('lbl_actual_a.npy')
     
    # pred_dist_b=np.load('pred_dist_b.npy')
    # lbl_dist_b=np.load('lbl_dist_b.npy')
    # lbl_actual_b=np.load('lbl_actual_b.npy')
    
    # mn_pred_a=np.load('mn_pred_a.npy')
    # mn_actual_a=np.load('mn_actual_a.npy')
    # mn_pred_b=np.load('mn_pred_b.npy')
    # mn_actual_b=np.load('mn_actual_b.npy')
    
    
    
    # sys.exit()
    
    
    
    # plt.style.use('plot_style')
    
    mycmap = 'Blues'
    
    aa = np.linspace(-5.0,-2.0,50)
    bb = np.linspace(-8.0,-3.0,50)
    x,y = np.meshgrid(bb,aa)
    
    
    
    def swap(line):
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        line.set_xdata(ydata)
        line.set_ydata(xdata)
    
    
    M=np.zeros([50,4])
    M[:,0]=pred_dist_a
    M[:,1]=lbl_dist_a
    M[:,2]=pred_dist_b
    M[:,3]=lbl_dist_b
    
    T=np.zeros([2,1])
    T[0]=lbl_actual_a
    T[1]=lbl_actual_b
    
    tmp=np.cumsum(M[:,0])
    tmp2=np.cumsum(M[:,1])
    
    mn_pred_a=np.argmin(np.abs(tmp-0.15))
    mx_pred_a=np.argmin(np.abs(tmp-0.85))
    err_var=mx_pred_a-mn_pred_a
    mn_gt_a=aa[np.min(np.where(M[:,1]>0))]
    mx_gt_a=aa[np.max(np.where(M[:,1]>0))]
    err_var2=mx_gt_a-mn_gt_a
    
    tmp3=np.cumsum(M[:,2])
    tmp4=np.cumsum(M[:,3])
    
    mn_pred_b=np.argmin(np.abs(tmp3-0.15))
    mx_pred_b=np.argmin(np.abs(tmp3-0.85))
    err_var=mx_pred_b-mn_pred_b
    mn_gt_b=bb[np.min(np.where(M[:,3]>0))]
    mx_gt_b=bb[np.max(np.where(M[:,3]>0))]
    err_var2=mx_gt_b-mn_gt_b
    
    
    
    
    
    
    # start with a square Figure
    fig = plt.figure(figsize=(8,5.6))
    
    # Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
    # the size of the marginal axes and the main axes in both directions.
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2,2,width_ratios=(7,2),height_ratios=(3,6),left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.05,hspace=0.05)
    #gs = fig.add_gridspec(2,2,left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.05,hspace=0.05)
    
    ax = fig.add_subplot(gs[1,0])
    ax_histx = fig.add_subplot(gs[0,0],sharex=ax)
    ax_histy = fig.add_subplot(gs[1,1],sharey=ax)
    
    
    
    
    # the scatter plot:
    # ax.set_xlabel(r'$\beta$')
    # ax.set_ylabel(r'log$_{10}(A)$')
    ax.set_xlabel(r'B')
    ax.set_ylabel(r'log(A)')
    
    ax.set_yticks([-5.0,-4.4,-3.8,-3.2,-2.6,-2.0])
    #ax.set_aspect('equal')
    plt.setp(ax.get_xticklabels()[-1],visible=False)
        
    a=np.outer(pred_dist_a[0,::-1],pred_dist_b[0,:])
    ax.imshow(a,cmap=mycmap,norm=mcolors.PowerNorm(0.4),alpha=0.8,extent=(min(bb),max(bb),min(aa),max(aa)))
    
    
    
    # sigmas  = [0.997,0.95,0.68]
    # clevels = [0,0,0]
    sigmas  = [0.997,0.95]
    clevels = [0,0]

    maxa = np.amax(a)
    thres = 0.0001*maxa
    dthres = 0.0001*maxa
    ind_sigma = 0
    while ind_sigma < 2:
        masked = np.ma.masked_where(a<thres,a)
        mysum = np.sum(masked)
        if mysum < sigmas[ind_sigma]:
            clevels[ind_sigma] = thres - dthres
            ind_sigma = ind_sigma + 1
            #print(thres,mysum,ind_sigma)
        thres = thres + dthres
    ax.contour(x,y[::-1],a,levels=clevels,colors=['b','b','b'],linestyles=['--','--','--'])
    
    
    
    rect=plt.Rectangle((mn_gt_b,mn_gt_a), mx_gt_b-mn_gt_b, mx_gt_a-mn_gt_a,fill=False,edgecolor='red',linewidth=1.7)
    ax.add_patch(rect)
    ax.scatter(min(bb) + mn_actual_b*np.ptp(bb)/50,min(aa) + mn_actual_a*np.ptp(aa)/50,color='red',marker='x',s=100)
    
    mysum = 0
    N = len(pred_dist_b[0,:])
    for i in range(0,N):
        mysum = mysum + bb[i]*pred_dist_b[0,i]
    mn_pred_b = mysum
    
    mysum = 0
    N = len(pred_dist_a[0,:])
    for i in range(0,N):
        mysum = mysum + aa[i]*pred_dist_a[0,i]
    mn_pred_a = mysum
    
    ax.scatter(mn_pred_b,mn_pred_a,color='blue',marker='x',s=100)
    
    
    ax.scatter(min(bb)+lbl_actual_b[0]*np.ptp(bb)/50,min(aa)+lbl_actual_a[0]*np.ptp(aa)/50,color='green',marker='*',s=150)
    
    
    ax_histx.plot(bb,pred_dist_b[0,:],color='blue',ls='--')
    ax_histx.plot(bb,lbl_dist_b,color='red',ls='-')
    #ax_histx.plot(lbl_actual_b[0,:],color='red',ls='--')
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.set_xlim(min(bb),max(bb))
    
    line, = ax_histy.plot(aa,pred_dist_a[0,:],color='blue',ls='--')
    swap(line)
    line, = ax_histy.plot(aa,lbl_dist_a,color='red',ls='-')
    swap(line)
    
    ax_histy.relim()
    ax_histy.autoscale_view()
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.set_ylim(min(aa),max(aa))
    
    plt.show()
    
    # plt.savefig("2d.png")
