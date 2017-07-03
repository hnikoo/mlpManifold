"""


"""
import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)
from sklearn.cross_validation import train_test_split

from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Dense, Activation,Lambda
from keras.optimizers import RMSprop, Adadelta, Adam
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomNormal
import keras.backend as K
from keras import objectives


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
    original_dim = 2
    latent_dim = 2
    batch_size = 256
    intermediate_dim = 256
    epsilon_std = 1.0
    
    x = Input(batch_shape=(batch_size,original_dim))
    h = Dense(intermediate_dim, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(x)
    z_mean = Dense(latent_dim,kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(h)
    z_log_var = Dense(latent_dim,kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))(h)
    
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size,latent_dim,), mean=0.,stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation='relu',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    decoder_mean = Dense(original_dim, activation='linear',kernel_initializer=RandomNormal(mean=0.0, stddev=0.01))
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    
    
    def vae_loss(x, x_decoded_mean):
        xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss    
    
    vae = Model(x, x_decoded_mean)
    repmodel = Model(x,z_mean)
    
    vae.compile(optimizer='rmsprop', loss=vae_loss)
    
    print vae.summary()
    return vae,repmodel


def train(model,repmodel,DATA):
    Xtrain,ytrain,Xval,yval,Xtest,ytest = DATA[:]
    Xtrain = (Xtrain - Xtrain.min()) / (Xtrain.max()-Xtrain.min())
    Xval = (Xval - Xtrain.min()) / (Xtrain.max()- Xtrain.min())
    
    #optimizer = Adadelta()
    #model.compile(loss='mae', optimizer=optimizer)
    
    NewManifolds = [Xtest]
    
    # train 
    print('training the model ...')
    for ep in xrange(500):
        model.fit(Xtrain, Xtrain,shuffle=True,
                    batch_size=256, nb_epoch=1,
                    verbose=1,validation_data=(Xval[:256,],Xval[:256,]))
        
        # get the new manifold of data representation
        if (ep % 1)==0:
            newRepXtest = repmodel.predict(Xtest[:256,],batch_size=256)
            NewManifolds.append(newRepXtest)
            
    return NewManifolds
    

def test(model,DATA):
    Xtrain,ytrain,Xval,yval,Xtest,ytest = DATA[:]
    Xtest = (Xtest - Xtrain.min()) /(Xtrain.max()- Xtrain.min())
    # test
    print('testing the model ...')
    yprd = model.predict(Xtest[:256,],batch_size=256)
    plt.scatter(yprd[:,0],yprd[:,1])
    plt.scatter(Xtest[:256,0],Xtest[:256,1])
    plt.show()
    
    print 'mean absolute error on Test set is: ', np.mean(np.abs(yprd.astype('float').flatten()-Xtest[:256,].flatten()))


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