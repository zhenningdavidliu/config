import h5py
import numpy as np
from numpy import linalg as LA 
import matplotlib.pyplot as plt
if __name__=='__main__':
    #first epoch
    fe = 500
    #last epoch
    le = 901
    ref_weights = h5py.File('model/7/weights_at_epoch-{}.h5'.format(fe),'r+') 
    
    A1 = np.array(ref_weights['dense_1']['dense_1']['kernel:0'])
    b1 = np.array(ref_weights['dense_1']['dense_1']['bias:0'])
    A2 = np.array(ref_weights['dense_2']['dense_2']['kernel:0'])
    b2 = np.array(ref_weights['dense_2']['dense_2']['bias:0'])
    
    print('|A1|: {}, |b1|: {}, |A2|: {}, |b2|: {}'.format(
                                            LA.norm(A1),
                                            LA.norm(b1),
                                            LA.norm(A2),
                                            LA.norm(b2)))
    
    for i in range(fe,le, 100):
        
        # Read current weights
        temp_weights = h5py.File('model/7/weights_at_epoch-{}.h5'.format(i),'r+') 
        
        A1_new = np.array(temp_weights['dense_1']['dense_1']['kernel:0']);
        b1_new = np.array(temp_weights['dense_1']['dense_1']['bias:0'])

        A2_new = np.array(temp_weights['dense_2']['dense_2']['kernel:0']);
        b2_new = np.array(temp_weights['dense_2']['dense_2']['bias:0'])

        # Compute relative difference 
        print("e: {0:04d}, |A1-A1n|/|A1|: {1:5e}, |b1-b1n|/|b1|: {2:5e}".format(i,
                                                LA.norm(A1-A1_new)/LA.norm(A1),
                                                LA.norm(b1-b1_new)/LA.norm(b1)))
        
        print("e: {0:04d}, |A2-A2n|/|A2|: {1:5e}, |b2-b2n|/|b2|: {2:5e}".format(i,
                                                LA.norm(A2-A2_new)/LA.norm(A2),
                                                LA.norm(b2-b2_new)/LA.norm(b2)))


    last_weights = h5py.File('model/7/weights_at_epoch-{}.h5'.format(le-1),'r+') 


    A1l = np.array(last_weights['dense_1']['dense_1']['kernel:0'])
    b1l = np.array(last_weights['dense_1']['dense_1']['bias:0'])
    A2l = np.array(last_weights['dense_2']['dense_2']['kernel:0'])
    b2l = np.array(last_weights['dense_2']['dense_2']['bias:0'])

    plt.figure();
    plt.subplot(131); plt.matshow(A1, fignum=False); plt.colorbar(); plt.title('A1')
    plt.subplot(132); plt.matshow(A1l, fignum=False); plt.colorbar(); plt.title('A1_last')
    plt.subplot(133); plt.matshow(abs(A1-A1l), fignum=False); plt.colorbar(); plt.title('|A1-A1_last|')
    plt.show();

