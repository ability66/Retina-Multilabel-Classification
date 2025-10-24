# Retina Multilabel Classification

本项目使用 **MONAI DenseNet121（ImageNet 预训练）** 做**多标签**眼底图像分类。训练脚本会在验证集上输出 AUC/F1，并在评测时将预测写回 CSV（多标签，按阈值逐列二值化）。

## 0) 目录结构
```
.
├── images/                # 放所有图片（文件名与 CSV 的 path 列一致）
├── train.csv              # 训练集 CSV
├── val.csv                # 验证集 CSV
├── test_sample.csv        # 测试/提交 CSV
└── main.py               # 训练/评估脚本
```
CSV 头部示例：
```
path,DR,NORMAL,MH,ODC,TSLN,ARMD,DN,MYA,BRVO,ODP,CRVO,CNV,RS,ODE,LS,CSR,HTR,ASR,CRS,OTHER
```
## 1) 准备 uv 虚拟环境
```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate
```

## 2) 安装 PyTorch
```
# CUDA 12.1:
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

## 3) 安装其余依赖
```
uv pip install -r requirements.txt
```
## 4) 训练
DenseNet121，多标签，BCEWithLogitsLoss + Sigmoid
``` python
python main.py \
  --data-dir ./images \
  --train-csv ./train.csv \
  --val-csv ./val.csv \
  --epochs 10 \
  --batch-size 16 \
  --lr 3e-4 \
  --img-size 512 \
  --model densenet121
```
训练期间会在验证集上打印：
 - AUC(macro), AUC(micro), F1(micro) 并把最优模型保存在 checkpoints/best.pt

## 5) 测试/评测
将最佳模型用于 test_sample.csv；多标签阈值默认 0.5
``` python
python main.py \
  --data-dir ./images \
  --val-csv ./test_sample.csv \
  --eval-only \
  --ckpt checkpoints/best.pt \
  --out-csv ./pred_test_sample.csv
```

输出 CSV 将保留原格式，并把各标签列改为预测的 0/1：
- 多标签：每一列独立阈值化（prob >= threshold -> 1，否则 0）
