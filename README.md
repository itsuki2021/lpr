# lpr
mmocr + torchserve，车牌识别

## 1.环境安装
```bash
pip install -r requirements.txt
```

## 2.训练模型
按mmocr官网训练textdet和textrecog模型，将*.pth模型转为*.mar文件，复制*.mar文件到[model-store](torchserve/model-store)中（略）


## 3.开启模型服务
```bash
cd shell
# torch-serve CPU版本
./create_container.sh
# 或者运行GPU版
./create_container_gpu.sh
```

## 4.运行Demo脚本
准备测试图像，确认配置文件[config.yaml](config/config.yaml)中的内容无误后，运行
```bash
python demo_torchserve.py <config-file>
```