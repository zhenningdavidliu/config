import numpy as np
from random_inputs import random_labeled_data

if __name__ == "__main__":
    
    data_size = 10000;
    randomness = 'gaussian'

    
    # set randomness to either 'gaussian', 'uniform' or 'stripes'
    data, labels, sums = random_labeled_data(data_size, randomness)

    #based on the labels, what decision would the NN make
    temp = np.rint(labels)
    #need to change dimensions because labels are (100,1)
    decision = [int(i) for i in list(zip(*temp))[0]]
    
    #whether the sum is <96 (1 if yes, 0 othervise)
    sum_cond = [1 if i<96 else 0 for i in sums]
       
    ''' 
    diff counts the number of elements that dont support our hypothesis 
    '''
    # diff = [sum(x) for x in zip(decision,np.negative(sum_cond)) ]
    diff = sum(1 for i in range(len(decision)) if decision[i] != sum_cond[i])
    
    #print(labels)
    print(np.amin(sums), np.amax(sums))
    
    '''
    count the number of elements which do not support the hypothesis 
    this is by counting the number of elements <90 for example that would be classified
    as 0 and the the ones that have >100 with classification 1
    '''

    fail = sum(1 for i in range(len(decision)) if (decision[i]!=sum_cond[i] and (sums[i]<90 or sums[i]>100 )))
    # fail_items = [sums[i] for i in range(len(decision)) if (decision[i]!=sum_cond[i] and (sums[i]<90 or sums[i]>100 ))]
    fail100 = sum(1 for i in range(len(decision)) if (decision[i]!=sum_cond[i] and sums[i]>100 ))
    fail90 =  sum(1 for i in range(len(decision)) if (decision[i]!=sum_cond[i] and sums[i]<90 ))

    
    print('The number of elements which disprove the 96 hypothesis: {}' .format(diff)) 
    print('The number of elements that disprove 96, with sum greater than 100: {}'.format(fail100))
    print('The number of elements that disprove 96, with sum less than 90: {}'.format(fail90))

    print('The number of labels of noise that the network thinks is 1: {} out of {}'.format(sum(decision), data_size))
