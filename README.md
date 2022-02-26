# lpr
车牌识别， 参考链接：[szad670401/HyperLPR](https://github.com/szad670401/HyperLPR)

## 1.环境安装
```bash
pip install -r requirements.txt
```

## 2.运行车牌识别脚本
修改[config.yaml](config/config.yaml)中的参数后运行detect.py脚本
```bash
python detect.py
```

## 3.MMOCR torch-serve
```bash
torchserve --start --ncs --model-store /home/model-server/model-store/ --models all
```
