# Chinese_correction_system

### step1 环境配置

运行以下指令安装依赖
```bash
pip install -r"requirements.txt"
```   

运行python文件从modelscope下载模型参数
```bash
python -u"build/get_require.py"
```

### step2 导入使用
```python
from Text_correct.correct import predict

original_text = "今天天器很不错"
correct_text = predict(original_text)

print(correct_text) # 输出 "今天天气很不错"

```
