# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import os
from scipy import interpolate
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
folder_name = './model'
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

##################   hyperparameters   #################################
lr=0.001
time_steps=1800 #number of sampling data for training
steps=2000
display_step=100
validation_steps=200 #number of sampling data for testing
filter_num = [1, 6, 6, 6, 1]
padding_size = [2, 2]
'''
input  Ret        filter_num      padding_size
       100          [1, 1]          [0,  9]
dwdy   180          [1, 1]          [0, 19]
       950          [1, 1]          [0, 19]

       100       [1, 6, 6, 1]       [1,  1]
dudy   180     [1, 6, 6, 6, 1]      [2,  2]        *taken for example
       950    [1, 6, 6, 6, 6, 1]    [4,  4]
'''
batch_size_v=1
N1Mf=16
N1M=128
N3M=128
LX=2*np.pi
LZ=np.pi
needreaddata = False

##################   load training data  #################################
if needreaddata:
    file1=open('VD10_T.dat')#-label
    file2=open('DUY_T.dat')#input
    '''
    In that case, the data is restored as:
    time1: x(:,1) x(:,2) ... x(:,N1M*N3M)
    time2: x(:,1) x(:,2) ... x(:,N1M*N3M)
    ...
    timeN: x(:,1) x(:,2) ... x(:,N1M*N3M)

    (The first dimension is the x-direction
        second                 z-direction)
    '''

    # Interpolation function from RNL to DNS
    def interr2d(data):
        x = np.arange(0, LZ, LZ/N3M)
        y = np.arange(0, LX, LX/N1Mf)
        z = np.reshape(data,[N1Mf,N3M])
        f = interpolate.interp2d(x,y,z,'linear')
        xnew = np.arange(0, LZ, LZ/N3M)
        ynew = np.arange(0, LX, LX/N1M)
        znew = f(xnew, ynew)
        data = np.squeeze(np.reshape(znew,[1,-1]))
        return data

    labels=[]
    inputs=[]
    for i in range(time_steps):
        file1.readline()
        file2.readline()
    for i in range(validation_steps):
        labels_data=file1.readline()
        labels_data=labels_data.split()
        inputs_data=file2.readline()
        inputs_data=inputs_data.split()
        for j in range(len(labels_data)):
            labels_data[j]=-float(labels_data[j])#the minus means that the label equations its opposite value
            inputs_data[j]=float(inputs_data[j])
        # Interpolation from RNL to DNS
        labels_data=interr2d(labels_data)
        inputs_data=interr2d(inputs_data)
        labels.append(labels_data)
        inputs.append(inputs_data)

    ##################   create batches   #################################
    batches=len(labels)//batch_size_v
    def createbatch(input_data,batches_num):
        inputs_batch=[]
        for i in range(batches_num):
            start_index=i*batch_size_v
            end_index=start_index+batch_size_v
            inputs=input_data[start_index:end_index]
            inputs_batch.append(inputs)
        return np.array(inputs_batch)
    batch_xs=createbatch(inputs,batches)
    batch_ys=createbatch(labels,batches)

    np.save('batch_xs_testing.npy',batch_xs)
    np.save('batch_xs_testing.npy',batch_ys)

else:
    batch_xs = np.load('batch_xs_testing.npy')
    batch_ys = np.load('batch_ys_testing.npy')
    batches = batch_xs.shape[0]

class PeriodicPadding(nn.Module):
    def __init__(self, padding_size):
        super(PeriodicPadding, self).__init__()
        self.nx = padding_size[0]
        self.nz = padding_size[1]

    def periodic_pad(self, X):
        if len(X.shape)<4:
            X=X[:,np.newaxis,:,:]
        if self.nx != 0:
            X = torch.cat([X[:,:,-self.nx:,:], X, X[:,:,0:self.nx,:]], dim=2)
        if self.nz != 0:
            X = torch.cat([X[:,:,:,-self.nz:], X, X[:,:,:,0:self.nz]], dim=3)
        return X
    
    def forward(self, x):
        return self.periodic_pad(x)

# Construct CNN
class CNN(nn.Module):
    def __init__(self, filter_num, padding_size):
        super(CNN, self).__init__()

        self.filter_num = filter_num
        self.layers_num = len(filter_num)-1
        self.kernel_size = (padding_size[0] * 2 + 1, padding_size[1] * 2 + 1)
        self.layers = nn.ModuleList()

    def initialize_layers(self):

        for i in range(self.layers_num):
            in_channels = self.filter_num[i]
            out_channels = self.filter_num[i+1]

            self.layers.append(nn.Sequential(
                PeriodicPadding(padding_size),
                nn.Conv2d(in_channels, out_channels, self.kernel_size, padding=0, stride=1, bias=False), 
                nn.BatchNorm2d(out_channels),
                nn.Tanh(), 
                ))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # make CNN
    filename = 'cnn' + str(1900)
    f = open('./model/'+filename+'.pickle', 'rb')
    cnn = pickle.load(f)
    f.close()
    cnn.eval()

    cnn.double().to(device)
    print(cnn)

    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr, eps=1e-8)

    ## quick look at the input and label
    plt.figure(figsize = (6.4, 9.6))
    dx=LX/N1M
    dz=LZ/N3M
    x = np.linspace(dx, LX, N1M)
    z = np.linspace(dz, LZ, N3M)
    X, Z = np.meshgrid(x, z)
    plt.subplot(2,1,1)
    plt.contourf(X,Z,np.reshape(batch_xs[0][0],(N1M,N3M)).T,100,cmap=cm.RdBu_r)
    plt.xlabel('x')
    plt.ylabel('z')
    cbar=plt.colorbar()
    cbar.set_label('input', size=18)
    plt.subplot(2,1,2)
    plt.contourf(X,Z,np.reshape(batch_ys[0][0],(N1M,N3M)).T,100,cmap=cm.RdBu_r)
    plt.xlabel('x')
    plt.ylabel('z')
    cbar=plt.colorbar()
    cbar.set_label('label', size=18)
    plt.savefig('input-label-contour-testing.png', dpi=300)
    plt.close()

    batch_xs = torch.tensor(batch_xs).to(device) # (batches, batch_size, N1M * N3M)
    batch_xs = batch_xs.reshape(-1, batch_size_v, N1M, N3M) 
    batch_ys = torch.tensor(batch_ys).to(device)
    batch_ys = batch_ys.reshape(-1, batch_size_v, N1M, N3M)
  
    # testing
    avg_cost=0.0
    avg_corr=0.0
    for batche in range(batches):
        output = cnn(batch_xs[batche])
        label = batch_ys[batche].unsqueeze(1)
        loss = (output - label) ** 2
        loss = (loss * (5.0 * abs(label)).exp()).sum() / 2.0 / batch_size_v

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():

            avg_cost=avg_cost+loss
            corr=np.mean([np.corrcoef(output[j].cpu().numpy().reshape(-1,N1M*N3M),
                                        batch_ys[batche][j].cpu().numpy().reshape(-1,N1M*N3M))[0,1] 
                                        for j in range(batch_size_v)])
        avg_corr=avg_corr+corr
        print('Epoch %d: loss = %f, corr = %f' % (batche, loss.item(), corr))

        file = open('Batch_loss_testing.plt','a')
        file.write(str(batche) + '  ' + str(loss.item()) + '  ' + str(corr) + '  ' + '\n')
        file.close()
                
    avg_cost = avg_cost / batches
    avg_corr = avg_corr / batches
    print("Loss = %f Corr = %f" % (avg_cost, avg_corr))
            
    torch.save(cnn.state_dict(), 'model.pkl')