import numpy as np
import os
import pickle

def load_data(train = True):
    def unpickle(filepath):
        with open(os.path.join('./cifar-10-batches-py', filepath), 'rb') as fo:
            dict_ = pickle.load(fo, encoding='bytes')
        return dict_

    if not os.path.exists('./cifar-10-batches-py'):
        raise ValueError('Need to run get_data.sh before writing any code!')

    full_data = None
    full_labels = None

    batches = [f'data_batch_{i+1}' for i in range(5)] if train else ['test_batch']
    for batch in batches:
        dict_ = unpickle(batch)
        data = np.array(dict_[b'data'].reshape(-1, 3, 1024).mean(axis=1))
        labels = np.array(dict_[b'labels']) 
        full_data = data if full_data is None else np.concatenate([full_data, data])
        full_labels = labels if full_labels is None else np.concatenate([full_labels, labels])

    return full_data, full_labels

def z_score_normalize(X, u = None, sd = None):
    """
    Performs z-score normalization on X. 

    f(x) = (x - μ) / σ
        where 
            μ = mean of x
            σ = standard deviation of x

    Parameters
    ----------
    X : np.array
        The data to z-score normalize
    u (optional) : np.array
        The mean to use when normalizing
    sd (optional) : np.array
        The standard deviation to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with mean 0 and stdev 1
            Computed statistics (mean and stdev) for the dataset to undo z-scoring.
    """
    # assume X represents array of a single image
    if(u==None):
        u = X.mean()
    if(sd==None):
        sd = X.std()
    X = (X-u)/sd
    return (X,u,sd)

def min_max_normalize(X, _min = None, _max = None):
    """
    Performs min-max normalization on X. 

    f(x) = (x - min(x)) / (max(x) - min(x))

    Parameters
    ----------
    X : np.array
        The data to min-max normalize
    _min (optional) : np.array
        The min to use when normalizing
    _max (optional) : np.array
        The max to use when normalizing

    Returns
    -------
        Tuple:
            Transformed dataset with all values in [0,1]
            Computed statistics (min and max) for the dataset to undo min-max normalization.
    """
    if(_min==None):
        _min = X.min() # a single value
    if(_max==None):
        _max = X.max()
    X = (X-_min)/(_max-_min)
    return (X,_min,_max)

def onehot_encode(y):
    """
    Performs one-hot encoding on y.

    Ideas:
        NumPy's `eye` function

    Parameters
    ----------
    y : np.array
        1d array (length n) of targets (k)

    Returns
    -------
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.
    """
    # list = []
    # for i in range y.shape():
    #     if(y[i] not in list):
    #         list.append(y[i])
    # k = len(list)
    k=10
    # res = np.zeros((y.shape,k))
    # for i in range (y.shape):
    #     res[i,y[i]] =1
    res = np.eye(k)[y]
    return res


def onehot_decode(y):
    """
    Performs one-hot decoding on y.

    Ideas:
        NumPy's `argmax` function 

    Parameters
    ----------
    y : np.array
        2d array (shape n*k) with each row corresponding to a one-hot encoded version of the original value.

    Returns
    -------
        1d array (length n) of targets (k)
    """
    # res = np.zeros(y.shape[0])
    # for i in range y.shape[0]:
    #     res[i] =
    return np.argmax(y,axis =1)

def shuffle(dataset):
    """
    Shuffle dataset.

    Make sure that corresponding images and labels are kept together. 
    Ideas: 
        NumPy array indexing 
            https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing

    Parameters
    ----------
    dataset
        Tuple containing
            Images (X)
            Labels (y)

    Returns
    -------
        Tuple containing
            Images (X)
            Labels (y)
    """
    X = dataset[0]
    # img = copy.deepcopy(X)
    y = dataset[1]
    # label = copy.deepcopy(y)
    length = len(X)
    #  or length = X.shape[0]
    arr = np.arange(length)
    np.random.shuffle(arr)
    # for i in range(length):
    #     img[i] = X[arr[i]]
    #     label[i] = y[arr[i]]
    return X[arr],y[arr]


def append_bias(X):
    """
    Append bias term for dataset.

    Parameters
    ----------
    X
        2d numpy array with shape (N,d)

    Returns
    -------
        2d numpy array with shape (N,d+1)
    """
    N, d = X.shape[0], X.shape[1]
    bias = np.array([np.ones(N)])
    return np.concatenate((X,bias.T),axis=1)

def generate_minibatches(dataset, batch_size=64):
    X, y = dataset
    l_idx, r_idx = 0, batch_size
    while r_idx < len(X):
        yield X[l_idx:r_idx], y[l_idx:r_idx]
        l_idx, r_idx = r_idx, r_idx + batch_size

    yield X[l_idx:], y[l_idx:]

def generate_k_fold_set(dataset, k = 10): 
    X, y = dataset
    if k == 1:
        yield (X, y), (X[len(X):], y[len(y):])
        return

    order = np.random.permutation(len(X))
    
    fold_width = len(X) // k

    l_idx, r_idx = 0, fold_width

    for i in range(k):
        train = np.concatenate([X[order[:l_idx]], X[order[r_idx:]]]), np.concatenate([y[order[:l_idx]], y[order[r_idx:]]])
        validation = X[order[l_idx:r_idx]], y[order[l_idx:r_idx]]
        yield train, validation
        l_idx, r_idx = r_idx, r_idx + fold_width
