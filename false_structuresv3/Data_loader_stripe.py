from Data_loader import Data_loader
import numpy as np

class Data_loader_stripe_train(Data_loader):
    """Data loader class for loading training data for the stripe example.

Images with a horizontal stripe, have a background colour of
`-a`, whereas the light stripe has value `1-a`. Images with a vertical
stripe have a background colour value of `a` and a light stripe with 
value `1+a`.

Attributes
----------
grid_size (int): Size of images (grid_size × grid_size)
a (float): Range of colours [-a, 1+a].
line_width (int): Number of pixels used for the line.
flip_a_values (bool): Whether to flip the sign of a, and hence the colour coding.

Methods
-------
load_data(): Loads the training data.
"""

    def __init__(self, arguments):
        super(Data_loader_stripe_train, self).__init__(arguments)    
        required_keys = ['grid_size', "line_width", "a", "flip_a_values"]
        
        self._check_for_valid_arguments(required_keys, arguments)
        self.grid_size = arguments['grid_size']
        self.line_width = arguments['line_width']
        self.a = arguments['a']
        self.flip_a_values = arguments['flip_a_values']

    def load_data(self):
        """ Load the stripe training data. Size of this data will be 
`N = 2*(n-line_width+1)`, and contain the unique elements one can create with a
fixed value for `a`. The images have the following colour codes

Horizontal line images:
    - Background colour: -a
    - Stripe colour: 1-a
Vertical line images:
    - Background colour: a
    - Stripe colour: 1+a

If the `flip_a_values` attribute is True, then `a` values will have opposite signs.

Returns
-------
data (ndarray): The image data of size (N, n,n) 
label (ndarray): Label of the images. Size (N,1). Horizontal stripe images have 
    label 1 and vertical stripe images have label 0. 
"""

        data, label = self._generate_all_combinations_of_stripe_images();

        return data, label;
    
    def _generate_all_combinations_of_stripe_images(self, shuffle=True):

        a = self.a;
        n = self.grid_size;
        line_width = self.line_width;
        flip_a_values = self.flip_a_values        
        
        sign = 1;
        if flip_a_values:
            sign = -sign;
        sign_hor = sign*(-1);
        sign_ver = sign*(1);

        lidx = np.arange(line_width);

        nbr_of_perm = n-line_width+1;
        data_hor = sign_hor*a*np.ones([nbr_of_perm, n, n]);
        data_ver = sign_ver*a*np.ones([nbr_of_perm, n, n]);
        label_hor = np.ones([nbr_of_perm, 1]);
        label_ver = np.zeros([nbr_of_perm, 1]);

        for i in range(nbr_of_perm):
            data_hor[i, i+lidx, :] += 1; 
            data_ver[i, i+lidx, :] += 1; 

        data = np.concatenate((data_hor, data_ver), axis=0);
        label = np.concatenate((label_hor, label_ver), axis=0);

        if shuffle:
            idx = np.arange(2*nbr_of_perm);
            np.random.shuffle(idx);
            data = data[idx];
            label = label[idx];

        p = n*line_width;
        if flip_a_values:
            for i in range(2*nbr_of_perm):
                s = np.sum(data[i,:,:]);
                if s > p and label[i] < 0.5:
                    print(i, 'Image wrongly classified as vertical')
                if s < p and label[i] > 0.5:
                    print(i, 'Image wrongly classified as horizontal, s: %g, p: %g, label[i]: %d' % (s,p,label[i]))
        else: 
            for i in range(2*nbr_of_perm):
                s = np.sum(data[i,:,:]);
                if s < p and label[i] < 0.5:
                    print(i, 'Image wrongly classified as vertical')
                if s > p and label[i] > 0.5:
                    print(i, 'Image wrongly classified as horizontal, s: %g, p: %g, label[i]: %d' % (s,p,label[i]))

        # change dimensions to 4

        data = np.expand_dims(data, axis = 3)

        return data, label;
        
    def __str__(self):
        class_str = """stripe data training
Grid size: %d
Line width: %d
a: %g
flip a values: %s
""" % (self.grid_size, self.line_width, self.a, self.flip_a_values)
        return class_str


class Data_loader_stripe_test(Data_loader):
    """Data loader class for loading test data for the stripe example.

Creates a dataset of size `data_size`, where each data point is an image with
either a horizontal stripe or a vertical stripe.  For each image a value x is
drawn from the interval [low_a_value, a] unsing a uniform distribution. Images with a
horizontal stripe, will then be given a background colour of `-x`, whereas the
light stripe has value `1-x`. Images with a vertical stripe will be given a
background colour value of `x` and a light stripe with value `1+x`. If the
`flip_a_values` attribute is True, the sing of `x` is flipped.

Attributes
----------
grid_size (int): Size of images (grid_size × grid_size)
a (float): Range of colours [-a, 1+a].
line_width (int): Number of pixels used for the line.
flip_a_values (bool): Whether to flip the sign of a, and hence the colour coding.
data_size (int): Number of images 
low_a_value (float): We pick x from the interval [low_a_value, a].

Methods
-------
load_data(): Loads the training data.
"""

    def __init__(self, arguments):
        super(Data_loader_stripe_test, self).__init__(arguments)    
        required_keys = ['grid_size', "line_width", "a", "flip_a_values", 
                         'data_size', 'low_a_value']
        
        self._check_for_valid_arguments(required_keys, arguments)
        self.grid_size = arguments['grid_size']
        self.line_width = arguments['line_width']
        self.a = arguments['a']
        self.flip_a_values = arguments['flip_a_values']
        print(self.flip_a_values)
        print(type(self.flip_a_values))
        self.data_size = arguments['data_size']
        self.low_a_value = arguments['low_a_value']

    def load_data(self):
        """ Load the stripe test data. 
Loads test data set of size `size`. The values of `x` will be drawn from 
a uniform distribution on the interval [low_a_value, a]. The images will have 
the following colour codes. 

Horizontal line images:
    - Background colour: -x
    - Stripe colour: 1-x
Vertical line images:
    - Background colour: x
    - Stripe colour: 1+x

If the `flip_a_values` attribute is True, then `x` values will have opposite signs.

Returns
-------
data (ndarray): The image data of size (N, n,n) 
label (ndarray): Label of the images. Size (N,1). Horizontal stripe images have 
    label 1 and vertical stripe images have label 0. 
"""

        data, label = self._generate_test_set();

        return data, label;

    def __str__(self):
        class_str = """stripe data testing
Grid size: %d
Line width: %d
a: %g
low a value: %g
flip a values: %s
data_size: %d
""" % (self.grid_size, self.line_width, self.a, self.low_a_value, 
       self.flip_a_values, self.data_size)
        return class_str

    def _generate_test_set(self):

        a = self.a
        n = self.grid_size
        line_width = self.line_width
        low_a_value = self.low_a_value
        flip_a_values = self.flip_a_values
        data_size = self.data_size

        stripe_idx = np.random.randint(low=0, high=n+1-line_width, size=data_size);
        a_values = np.random.uniform(low=low_a_value, high=a, size=data_size);
        is_horizontal = np.random.choice(a=[True, False], size=data_size, p=[0.5, 0.5])
        lidx = np.arange(line_width);

        data = np.zeros([data_size, n, n]);
        label = np.zeros([data_size, 1]);
        for i in range(data_size):

            if is_horizontal[i]:
                sign = -1;
            else:
                sign = 1;

            if flip_a_values:
                sign = -sign;

            data[i, :, :] = sign*a_values[i]; 

            if is_horizontal[i]:
                data[i, stripe_idx[i]+lidx, :] += 1; 
                label[i] = 1;
            else:
                data[i, :, stripe_idx[i]+lidx] += 1; 
                label[i] = 0;

        p = self.line_width*n;
        if flip_a_values:
            for i in range(data_size):
                s = np.sum(data[i,:,:]);
                if s > p and label[i] < 0.5:
                    print(i, 'Image wrongly classified as vertical')
                if s < p and label[i] > 0.5:
                    print(i, 'Image wrongly classified as horizontal, s: %g, p: %g, label[i]: %d' % (s,p,label[i]))

        else:
            for i in range(data_size):
                s = np.sum(data[i,:,:]);
                #print('s < p: %6s, s: %g, p: %g, label[i]: %d' % (s<p,s,p,label[i]))
                # s < p implies horizontal, label for horizontal is 1
                # s > p implies vertical, label for vertical is 0
                if s < p and label[i] < 0.5:
                    print(i, 'Image wrongly classified as vertical')
                if s > p and label[i] > 0.5:
                    print(i, 'Image wrongly classified as horizontal, s: %g, p: %g, label[i]: %d' % (s,p,label[i]))

        # make the data a 4 dimensional tensor for Neural Networks
        data = np.expand_dims(data, axis = 3)

        return data, label;


