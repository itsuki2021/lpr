# lpr
车牌识别， 参考链接：[szad670401/HyperLPR](https://github.com/szad670401/HyperLPR)

## 1.环境安装
```bash
pip install -r requirements.txt
```

## 2.运行车牌识别脚本
修改[detect.py](detect.py)中的参数
```bash
python detect.py
```

## 3.训练
[armaab/hyperlpr-train ](https://github.com/armaab/hyperlpr-train)
链接中模型保存格式为tensorflow默认格式（saved_model），并提供了.h5格式转换，但与推理代码中的模型格式不符（caffe model），
训练结果的使用方式未知， 初步判定训练脚本不可用
