# 主干网络
主目录 .\backbone
使用 [[pytorch-lightning]](https://lightning.ai/docs/pytorch/latest/) + [[PPQ]](https://github.com/openppl-public/ppq)
### 分析网络参数
```
python lightning_wrapper.py
```
### 训练并验证
```
python trainer.py --model=[NETTYPE] --width_mult=[WIDTH_MULT] --accelerator=gpu --devices=1 --max_epochs=60 --batch_size=256 --load_last
# 例如 训练 ghostnet
python trainer.py  --model=GhostNet --width_mult=0.6 --accelerator=gpu --devices=1 --max_epochs=60 --batch_size=256 --load_last
```
### 量化损失分析
注意，由于不同版本的pytorch导出onnx的node的名称不同，这里限定使用 pytorch 1.10.1
```
python result_plot.py
```

# 孪生网络目标跟踪网络
主目录 .\tracking 
参考 [[pytracking]](https://github.com/visionml/pytracking)
### 文件目录结构
.\ltr 模型与训练
.\ltr\onnx
.\ltr\onnx ONNX导出
.\ltr\calibration 校准数据集
.\ltr\pretrained_networks 对比方法的checkpoint
.\pytracking 测试与验证
.\pytracking\misctools 实用工具
.\pytracking\tracking_results 测试结果

### 训练
```
python .\ltr\run_training.py qfnet [CONFIG]
# 例如 ghostnet+attention+heatmap
python .\ltr\run_training.py qfnet qfattn
```
### 生成校准数据集
```
python .\ltr\run_training.py qfnet calibration
```
### 可视化调试
```
python .\pytracking\run_tracker.py qfnet [CONFIG] --dataset_name DATASET --sequence SEQUENCE --debug 10
```
### 运行
```
python .\pytracking\run_tracker.py qfnet [CONFIG] --dataset_name DATASET --threads THREAD
# CONFIG: qfattn qfconcat ...
```
### 量化后运行
```
python .\pytracking\run_tracker.py qfnet [CONFIG] --dataset_name DATASET --threads THREAD
# CONFIG: qfattn_q qfconcat_q ...
```
### 对比方法
[[SiamRPN(PYSOT)]](https://github.com/STVIR/pysot)
```
# Alexnet 量化前
python .\pytracking\run_tracker.py siamrpn siamrpn_alex --dataset_name DATASET --threads THREAD
# Alexnet 量化后
python .\pytracking\run_tracker.py siamrpn siamrpn_alex_q --dataset_name DATASET --threads THREAD
# Mobilenetv2 量化前
python .\pytracking\run_tracker.py siamrpn siamrpn_mobilenetv2 --dataset_name DATASET --threads THREAD
# Mobilenetv2 量化后
python .\pytracking\run_tracker.py siamrpn siamrpn_mobilenetv2_q --dataset_name DATASET --threads THREAD
```
[[LightTrack]](https://link.zhihu.com/?target=https%3A//github.com/researchmm/LightTrack)
```
# 量化前
python .\pytracking\run_tracker.py lighttrack default --dataset_name DATASET --threads THREAD
# 量化后
python .\pytracking\run_tracker.py lighttrack quant --dataset_name DATASET --threads THREAD
```
[[STARK-Lightning]](https://github.com/researchmm/Stark)
```
# 量化前
python .\pytracking\run_tracker.py starklightning default --dataset_name DATASET --threads THREAD
# 量化后
python .\pytracking\run_tracker.py starklightning quant --dataset_name DATASET --threads THREAD
```

### 显示结果
```
.\pytracking\misctools\analysis_results.py
```

### 量化数据显示
注意，由于不同版本的pytorch导出onnx的node的名称不同，这里限定使用 pytorch 1.10.1
```
# 显示特征融合网络的量化结果分析
python .\pytracking\misctools\fusionmethod.py
# 显示分支预测结构的量化结果分析
python .\pytracking\misctools\reghead.py
# 显示不同跟踪算法的量化结果分析
python .\pytracking\misctools\tracker.py
```

### 端侧部署
查看[[PPQ]](https://github.com/openppl-public/ppq)相关帮助
