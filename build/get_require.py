"""用于从外部获取相关依赖资源"""

import os
import shutil
import requests
from utils import visualize_download

def download_model_from_modelscope(author, model_name, save_dir="./"):
    from modelscope.hub.snapshot_download import snapshot_download
    '''
    从modelscope下载预训练模型
    '''
    if not os.path.exists(os.path.join(author, model_name)):
        snapshot_download(os.path.join(author, model_name), cache_dir=save_dir)

    shutil.move(os.path.join(author, model_name), model_name)
    shutil.rmtree(author)
    shutil.rmtree("temp")

def download_SIGHAN():
    '''
    从modelscope下载我们上传的数据集json文件
    '''
    train_url = 'https://modelscope.cn/api/v1/datasets/heaodong/SIGHAN2015/repo?Revision=master&FilePath=train.json'
    dev_url = 'https://modelscope.cn/api/v1/datasets/heaodong/SIGHAN2015/repo?Revision=master&FilePath=dev.json'
    test_url = 'https://modelscope.cn/api/v1/datasets/heaodong/SIGHAN2015/repo?Revision=master&FilePath=test.json'
    train_resp = requests.get(train_url, stream=True)
    dev_resp = requests.get(dev_url, stream=True)
    test_resp = requests.get(test_url, stream=True)
    if not os.path.exists('SIGHAN2015'):
        os.mkdir('SIGHAN2015')

    visualize_download(train_resp, "SIGHAN2015/train.json", "train", 99038414)
    visualize_download(dev_resp, "SIGHAN2015/dev.json", "dev", 11057130)
    visualize_download(test_resp, "SIGHAN2015/test.json", "test", 346770)

if __name__ == "__main__":
    download_model_from_modelscope("heaodong", "DecBert")
    # download_SIGHAN()
