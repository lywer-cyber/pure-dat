# DAT 数值全息超分辨率适配

本目录保留官方 DAT（Dual Aggregation Transformer）主干结构，并增加了单通道全息强度 ×4 的数据、训练和测试适配。网络接收归一化到 `[0,1]` 的单通道 LR 强度图，输出对应 HR 强度图；不做 RGB/Y 通道转换，也不加载官方 RGB 预训练权重。

## 数据格式

默认数据根目录为 `/root/autodl-tmp/real_hologram_x4`（与本项目现有 DAT 全息实验一致）；如果数据在其他位置，请同步修改训练和测试 YAML 中的 `dataroot`。目录结构为：

```text
real_hologram_x4/
  manifest.json
  train/{input,hologram_clean,metadata}/
  val/{input,hologram_clean,metadata}/
  test/{input,hologram_clean,metadata}/
```

每个 `.npy` 样本必须是 `[H,W]` 或 `[1,H,W]` 的有限浮点强度数组，范围 `[0,1]`；输入/目标文件名与 metadata JSON 文件名需要一一对应，HR 尺寸必须是 LR 的 4 倍。`manifest.json` 用于校验 schema、任务、scale、协议和 split 数量。默认使用 `lab_like` 协议下 `hologram_clean` 作为目标。

数据适配器为 `basicsr/data/hologram_intensity_dataset.py`。它只对训练样本做成对随机裁剪、翻转和转置；验证/测试保持原图。验证指标按未量化的浮点强度计算 PSNR、SSIM、MAE、RMSE 和频谱指标，避免先转 8-bit PNG 再评价。

## 训练与测试

在 DAT 仓库根目录安装依赖，然后运行：

```bash
pip install -r requirements.txt
python setup.py develop
python basicsr/train.py -opt options/Train/train_DAT_holo_x4.yml --launcher none
python basicsr/test.py -opt options/Test/test_DAT_holo_x4.yml
```

训练配置为单通道 DAT、×4、L1 损失、batch size 2、固定 seed 2026、25,500 iter；数据集 1,700 张训练图时每 850 iter 约为一个 epoch。DAT 的通道注意力中有 BatchNorm，因此训练 batch size 不应小于 2。测试配置默认读取训练产生的 EMA 权重 `experiments/DAT_hologram_x4_lab_like_25500/models/net_g_latest.pth`。

数据实际放在其他位置时，同时修改两个 YAML 中的 `dataroot`。路径相对当前工作目录解析，因此建议从 DAT 仓库根目录启动命令。训练/测试权重与实验输出会写入 `experiments/`、`results/`；源码中不包含全息数据或预训练检查点。
