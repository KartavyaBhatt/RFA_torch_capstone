import json
import os
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

# from .language_utils import word_to_indices, letter_to_vec, \
#     bag_of_words, get_word_emb_arr, val_to_vec, split_line, \
#     letter_to_idx

# TODO: capitalize global vars names and initialize to None

VOCAB_DIR = 0
emb_array = 0
vocab = 0
embed_dim = 0


def batch_data(data, batch_size, rng=None, shuffle=True, eval_mode=False, full=False):
    """
    data is a dict := {'x': [list], 'y': [list]} with optional fields 'y_true': [list], 'x_true' : [list]
    If eval_mode, use 'x_true' and 'y_true' instead of 'x' and 'y', if such fields exist
    returns x, y, which are both lists of size-batch_size lists
    """
    x = data['x_true'] if eval_mode and 'x_true' in data else data['x']
    y = data['y_true'] if eval_mode and 'y_true' in data else data['y']
    raw_x_y = list(zip(x, y))
    if shuffle:
        assert rng is not None
        rng.shuffle(raw_x_y)
    raw_x, raw_y = map(list, zip(*raw_x_y))
    batched_x, batched_y = [], []
    if not full:
        for i in range(0, len(raw_x), batch_size):
            batched_x.append(raw_x[i:i + batch_size])
            batched_y.append(raw_y[i:i + batch_size])
    else:
        batched_x.append(raw_x)
        batched_y.append(raw_y)
    return batched_x, batched_y


def read_data(train_data_dir, test_data_dir, split_by_user=True, dataset="femnist"):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    if dataset == 'sent140':
        global VOCAB_DIR
        global emb_array
        global vocab
        global embed_dim
        VOCAB_DIR = 'sent140/embs.json'
        emb_array, _, vocab = get_word_emb_arr(VOCAB_DIR)
        # print('shape obtained : ' + str(emb_array.shape))
        embed_dim = emb_array.shape[1]

    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    # START Old version :
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('reading train file ' + str(file_path))
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        print('reading test file ' + str(file_path))
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])
    # END Old version

    # counter = 0
    # for f in train_files:
    #    file_path = os.path.join(train_data_dir, f)
    #    with open(file_path, 'r') as inf:
    #        cdata = json.load(inf)
    #    clients.extend(cdata['users'])
    #    if 'hierarchies' in cdata:
    #        groups.extend(cdata['hierarchies'])
    #    train_data.update(cdata['user_data'])
    #    counter += 1
    #    if counter == 50:
    #        break

    # clients = [list(train_data.keys()). list(test_data.keys())]
    if split_by_user:
        clients = {
            'train_users': list(train_data.keys()),
            'test_users': list(test_data.keys())
        }
    else:
        clients = {
            'train_users': list(train_data.keys())
        }

    return clients, groups, train_data, test_data


def read_data_new(train_data_dirs, test_data_dirs):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    clientIDs = [0, 1, 2, 5, 10, 20]
    train_data = []
    test_data = []
    train_data_dirs = ['/home/kb8077/RFA_torch_capstone/data/traffic_sign/train/',
                 '/home/kb8077/RFA_torch_capstone/data/traffic_sign/validation-distorted/buffer128loss1/',
                 '/home/kb8077/RFA_torch_capstone/data/traffic_sign/validation-distorted/buffer128loss2/',
                 '/home/kb8077/RFA_torch_capstone/data/traffic_sign/validation-distorted/buffer128loss5/',
                 '/home/kb8077/RFA_torch_capstone/data/traffic_sign/validation-distorted/buffer256loss10/',
                 '/home/kb8077/RFA_torch_capstone/data/traffic_sign/validation-distorted/buffer512loss20/']

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for train_data_dir in tqdm(train_data_dirs, desc='Setting up clients'):
        train_class_dirs = os.listdir(train_data_dir)
        imgs = []
        labels = []
        for d in train_class_dirs:
            train_files = os.listdir(os.path.join(train_data_dir, d))
            train_files = [f for f in train_files if f.endswith('.jpg')]

            for f in train_files:
                file_path = os.path.join(train_data_dir, d, f)
                x = Image.open(file_path)
                try:
                    np.array(x)
                    imgs.append(preprocess(x))
                except OSError as error:
                    continue
                labels.append(1 if d == 'stop_signs' else 0)
        X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.33, random_state=42)
        train_data.append({'x': X_train, 'y': y_train})
        test_data.append({'x': X_test, 'y': y_test})


    # test_class_files = os.listdir(test_data_dir)
    # for d in test_class_files:
    #     test_files = os.listdir(os.path.join(test_data_dir, d))
    #     test_files = [f for f in test_files if f.endswith('.jpg')]
    #     for f in tqdm(test_files, desc='Reading test data'):
    #         file_path = os.path.join(test_data_dir, d, f)
    #         x = Image.open(file_path)
    #         test_data['x'].append(np.array(x))
    #         test_data['y'].append(1 if d == 'stop_signs' else 0)

    return clientIDs, train_data, test_data


def preprocess_data_x(list_inputs, dataset='femnist', center=False,
                      model_name=None):
    if dataset == 'femnist':
        if center:
            res = np.array(list_inputs) - np.tile(np.mean(list_inputs, axis=0), (len(list_inputs), 1))  # center data
            res = res.tolist()
            return res
        else:
            return list_inputs

    elif dataset == 'shakespeare':
        formatted_list_inputs = shakespeare_preprocess_x(list_inputs)
        return formatted_list_inputs

    elif dataset == 'sent140':
        return sent140_preprocess_x(list_inputs).tolist()


def preprocess_data_y(list_labels, dataset='femnist', model_name=None):
    assert model_name is not None
    if dataset == 'femnist' and ('cnn' in model_name):
        # return labels as is
        return list_labels
        # return femnist_preprocess_y_int(list_labels)
    elif dataset == 'femnist':  # one hot preprocess
        return femnist_preprocess_y_onehot(list_labels)
    elif dataset == 'shakespeare':
        return shakespeare_preprocess_y(list_labels)
    elif dataset == 'sent140':
        return sent140_preprocess_y(list_labels)


def femnist_preprocess_y_onehot(raw_y_batch):
    res = []
    for i in range(len(raw_y_batch)):
        num = np.zeros(62)  # Number of classes
        num[raw_y_batch[i]] = 1.0
        res.append(num)
    return res


def shakespeare_preprocess_x(raw_x_batch):
    x_batch = [[letter_to_idx(l) for l in x] for x in raw_x_batch]
    return x_batch


def shakespeare_preprocess_y(raw_y_batch):
    y_batch = [letter_to_idx(c) for c in raw_y_batch]
    return y_batch


def sent140_preprocess_x(X):
    x_batch = [e[4] for e in X]  # list of lines/phrases
    x = np.zeros((len(x_batch), embed_dim))
    for i in range(len(x_batch)):
        line = x_batch[i]
        words = split_line(line)
        idxs = [vocab[word] if word in vocab.keys() else emb_array.shape[0] - 1
                for word in words]
        word_embeddings = np.mean([emb_array[idx] for idx in idxs], axis=0)
        x[i, :] = word_embeddings
    return x


def sent140_preprocess_y(raw_y_batch):
    res = []
    for i in range(len(raw_y_batch)):
        res.append(float(raw_y_batch[i]))
    return res

# if __name__ == '__main__':
#     trainDirs = ['/home/kb8077/RFA_torch_capstone/data/traffic_sign/train/',
#                  '/home/kb8077/RFA_torch_capstone/data/traffic_sign/validation-distorted/buffer128loss1/',
#                  '/home/kb8077/RFA_torch_capstone/data/traffic_sign/validation-distorted/buffer128loss2/',
#                  '/home/kb8077/RFA_torch_capstone/data/traffic_sign/validation-distorted/buffer128loss5/',
#                  '/home/kb8077/RFA_torch_capstone/data/traffic_sign/validation-distorted/buffer256loss10/',
#                  '/home/kb8077/RFA_torch_capstone/data/traffic_sign/validation-distorted/buffer512loss20/']
#     read_data_new(trainDirs, '/home/kb8077/RFA_torch_capstone/data/traffic_sign/validation/')
