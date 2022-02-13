# lpr
车牌识别

## 1.环境安装
```bash
pip install -r requirements.txt
```
安装完毕后需要修改hyperlpr源码中的两个小bug
* https://github.com/szad670401/HyperLPR/issues/318
* https://github.com/szad670401/HyperLPR/issues/334

## 2.运行车牌识别脚本
将测试图像放入[data/](data)文件夹中
```bash
python demo_lpr.py
```
输出结果将保存至output文件夹

## 3.训练
待完成