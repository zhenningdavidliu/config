import h5py

f = h5py.File('model/7/keras_model_files.h5','r')
dset = f['model_weights']['dense_1']['dense_1']['kernel:0']
dset_bias = f['model_weights']['dense_1']['dense_1']['bias:0']
print(dset[:])
print(dset_bias[:])


