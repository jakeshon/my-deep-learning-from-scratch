try:
    import urllib.request
except ImportError:
    print("You should use Python 3.x")
import os
import pickle
import gzip
import numpy as np
from PIL import Image

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath("__file__"))
save_file = dataset_dir + "/mnist.pkl"

img_size = 784

# numpy 형태로 바꾼다.
def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])
    return dataset

# 이미지를 읽어서 numpy로 돌려줌
def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name
    print("Converting image "+ file_name + " to numpy array..")
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    
    data = data.reshape(-1, img_size)
    print("Done. shape=" + str(data.shape))
    return data

# 라벨을 읽어서 numpy로 돌려줌
def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name
    print("Converting label "+ file_name + " to numpy array..")
    with gzip.open(file_path, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    
    print("Done")
    return data

# 초기화
def init_mnist():
    #다운로드한다.
    download_mnist()
    #numpy형태로 바꾼다.
    dataset = _convert_numpy()

    #저장한다.
    print("Save " + save_file)
    with open(save_file, "wb") as f:
        pickle.dump(dataset, f, -1)
    print("Done")

def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    print(file_path)
    #file_path = file_path.replace("\\","/")
    if os.path.exists(file_path):
        return
    
    print("Downloding remote : " + url_base + file_name + " -> local : " + file_path)
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")


# 다운로드
def download_mnist():
    for file in key_file.values():
        _download(file)


# Load mnist data
def load_mnist():
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, "rb") as f:
        dataset = pickle.load(f)
    
    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


def img_show(img):
    pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img.show()

(x_train, t_label), (x_test, t_test) = load_mnist()
x_train.shape
x_test.shape

img = x_train[0]
img = img.reshape(28,28)

img_show(img)