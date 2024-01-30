import os
import shutil
from modelscope.hub.snapshot_download import snapshot_download

if os.path.exists("dienstag/chinese-bert-wwm/"):
    shutil.move("dienstag/chinese-bert-wwm/", 'chinese-bert-wwm/')

def download_model_from_modelscope(author, model_name, save_dir="./"):
    """
    从modelscope下载预训练模型
    """
    if not os.path.exists(os.path.join(author, model_name)):
        snapshot_download(os.path.join(author, model_name), cache_dir=save_dir)

    shutil.move(os.path.join(author, model_name), model_name)
    shutil.rmtree(author)
    shutil.rmtree("temp")

if __name__ == "__main__":
    download_model_from_modelscope("dienstag", "chinese-bert-wwm")