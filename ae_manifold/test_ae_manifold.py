"""
A simple script to train auto encoder and visualize the change of manifold of data over training.


"""
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)
from sklearn.cross_validation import train_test_split

from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, Activation
from keras.optimizers import RMSprop, Adadelta, Adam
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomNormal


def gen_data(): 
    # generate data ....
    print('Genete data ...')
    X = np.arange(0, 2, 0.001)
    Y1 = 0.5 + np.sin(np.pi * X)
    Y2 = 1.0 + np.sin(np.pi * X) 
    XX = np.concatenate((X, X), axis=0)
    YY = np.concatenate((Y1,Y2),axis=0)
    XX = np.concatenate((np.expand_dims(XX,axis=1),np.expand_dims(YY,axis=1)),axis=1)
    YY = np.concatenate((np.zeros(X.shape[0]),np.ones(X.shape[0])),axis=0)
    idx = np.arange(XX.shape[0])
    idx_train, idx_test = train_test_split(idx, test_size=0.2)
    idx_train,idx_val = train_test_split(idx_train,test_size=0.2)
    Xtrain = XX[idx_train,]
    ytrain = YY[idx_train,]
    Xval = XX[idx_val,]
    yval = YY[idx_val,]
    Xtest = XX[idx_test,]
    ytest = YY[idx_test,]
    return Xtrain,ytrain,Xval,yval,Xtest,ytest


def mlp_model():
    # define mlp model
    print('Build model...')
    inL = Input(shape=(2,))
    x = Dense(100,activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(inL)
    x = Dense(2,activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    x2 = Dense(100,activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    out = Dense(2,activation='linear',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x2)
    
    model = Model(inL,out)
    repmodel = Model(inL,x)
    
    print model.summary()
    
    return model,repmodel


def train(model,repmodel,DATA):
    Xtrain,ytrain,Xval,yval,Xtest,ytest = DATA[:]
    optimizer = Adadelta()
    model.compile(loss='mae', optimizer=optimizer)
    
    NewManifolds = [Xtest]
    
    # train 
    print('training the model ...')
    for ep in xrange(500):
        model.fit(Xtrain, Xtrain,shuffle=True,
                    batch_size=256, nb_epoch=1,
                    verbose=1,validation_data=(Xval,Xval))
        
        # get the new manifold of data representation
        if (ep % 1)==0:
            newRepXtest = repmodel.predict(Xtest,batch_size=128)
            NewManifolds.append(newRepXtest)
            
    return NewManifolds
    

def test(model,DATA):
    Xtrain,ytrain,Xval,yval,Xtest,ytest = DATA[:]
    # test
    print('testing the model ...')
    yprd = model.predict(Xtest,batch_size=128)
    plt.scatter(yprd[:,0],yprd[:,1])
    plt.scatter(Xtest[:,0],Xtest[:,1])
    plt.show()
    
    print 'mean absolute error on Test set is: ', np.mean(np.abs(yprd.astype('float').flatten()-Xtest.flatten()))


def visualize_manifold(manifolds,DATA):
    Xtrain,ytrain,Xval,yval,Xtest,ytest = DATA[:]
    idx1 = np.where(ytest==0)[0]
    idx2 = np.where(ytest==1)[0]
    for i in xrange(len(manifolds)):
        newRepXX = manifolds[i]
        plt.scatter(newRepXX[idx1,0],newRepXX[idx1,1])
        plt.scatter(newRepXX[idx2,0],newRepXX[idx2,1])
        #plt.savefig('./frames/'+str(i)+'.png')
        plt.pause(0.1)
        plt.clf()
        
    plt.show()
    

if __name__=='__main__':
    DATA = gen_data()
    model,repmodel = mlp_model()
    manifolds = train(model,repmodel,DATA)
    test(model,DATA)
    visualize_manifold(manifolds,DATA)