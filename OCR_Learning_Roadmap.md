# OCR 学习路线图：从 CNN 到 Transformer

> 三站式实战学习——每站都有代码、有调试、有可视化。
> 目标：直观理解"特征"在深度学习里到底是什么。

---

## 环境总览

```bash
# 三站通用依赖（推荐 Python 3.10+）
pip install torch torchvision matplotlib pillow tqdm numpy
# 第2站额外：合成 OCR 数据
pip install trdg
# 第3站额外：HuggingFace Transformers
pip install transformers sentencepiece
```

| 设备建议 | 说明 |
|---------|------|
| MNIST / 小 CRNN | CPU 可跑，几分钟出结果 |
| 大型 CRNN / TrOCR | 建议有 CUDA GPU，否则推理较慢 |

---

# 第1站：CNN 特征提取 —— MNIST 手写数字

## 1.1 学习目标

- 知道卷积层、池化层、全连接层各自干什么
- 用 `register_forward_hook` 抓出任意中间层的输出
- **直观看到**：浅层提取边缘/纹理，深层提取抽象语义

## 1.2 核心代码

### 模型定义 + 训练 (`stage1_cnn_mnist.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------- 超参 ----------
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001

# ---------- 数据 ----------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=1000, shuffle=False)

# ---------- 模型 ----------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------- 训练 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
    print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | Acc: {correct/100:.2f}%")

torch.save(model.state_dict(), "stage1_cnn_mnist.pth")
print("Model saved: stage1_cnn_mnist.pth")
```

### 特征可视化脚本 (`stage1_visualize.py`)

```python
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
from stage1_cnn_mnist import SimpleCNN

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model = SimpleCNN()
model.load_state_dict(torch.load("stage1_cnn_mnist.pth", map_location="cpu"))
model.eval()

model.conv1.register_forward_hook(get_activation("conv1"))
model.conv2.register_forward_hook(get_activation("conv2"))
model.conv3.register_forward_hook(get_activation("conv3"))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
img, label = ds[0]
img_batch = img.unsqueeze(0)

with torch.no_grad():
    _ = model(img_batch)

fig, axes = plt.subplots(4, 1, figsize=(16, 10))
axes[0].imshow(img.squeeze(), cmap="gray")
axes[0].set_title(f"Original (label={label})", fontsize=14)
axes[0].axis("off")

layer_names = ["conv1", "conv2", "conv3"]
for i, name in enumerate(layer_names):
    feat = activations[name][0]
    n_ch = min(feat.size(0), 16)
    grid = torch.cat([feat[j].unsqueeze(0) for j in range(n_ch)], dim=0)
    grid = torch.cat([grid[j] for j in range(n_ch)], dim=1)
    axes[i+1].imshow(grid.cpu().numpy(), cmap="viridis")
    axes[i+1].set_title(f"{name} - first {n_ch} channels", fontsize=13)
    axes[i+1].axis("off")

plt.tight_layout()
plt.savefig("stage1_feature_maps.png", dpi=150)
plt.show()
print("Saved: stage1_feature_maps.png")
```

### 运行命令

```bash
python stage1_cnn_mnist.py      # train (CPU ~2 min)
python stage1_visualize.py      # output stage1_feature_maps.png
```

### 调试检查清单

| 检查项 | 预期结果 |
|--------|---------|
| conv1 特征图 | 边缘轮廓清晰（数字笔画高频信息） |
| conv2 特征图 | 边缘模糊，出现纹理/局部形状 |
| conv3 特征图 | 高度抽象，人眼难辨认但分类器靠它决策 |
| 测试准确率 | > 98%（5 epoch 后） |

---

# 第2站：时序特征 —— CRNN OCR

## 2.1 学习目标

- 理解 CNN 为何直接出文字会乱序
- 理解 LSTM/RNN 如何赋予时序建模能力
- 理解 CTC Loss 如何解决对齐问题
- **亲自动手对比**：同一代码，去掉 LSTM 看效果差异

## 2.2 用 trdg 生成合成数据

```bash
trdg -c 5000 -w 5 -f 64 --output_dir data/ocr/train -tc "#000000,#888888"
trdg -c 1000 -w 5 -f 64 --output_dir data/ocr/test  -tc "#000000,#888888"
```

## 2.3 核心代码: 完整 CRNN (`stage2_crnn_ocr.py`)

```python
import os, string, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

IMG_H, IMG_W = 32, 128
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
CHAR_SET = string.ascii_lowercase + string.digits + " "
CHAR2IDX = {c: i+1 for i, c in enumerate(CHAR_SET)}
IDX2CHAR = {i+1: c for i, c in enumerate(CHAR_SET)}
IDX2CHAR[0] = "-"
NUM_CLASS = len(CHAR_SET) + 1

def load_ocr_data(folder):
    samples = []
    for fname in os.listdir(folder):
        if fname.endswith((".jpg", ".png", ".jpeg")):
            label = fname.rsplit("_", 1)[0].lower()
            label = "".join(c for c in label if c in CHAR_SET)
            if len(label) > 0:
                samples.append((os.path.join(folder, fname), label))
    return samples

class OCRDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((IMG_H, IMG_W)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        encoded = torch.tensor([CHAR2IDX[c] for c in label], dtype=torch.long)
        return img, encoded, len(label)

def collate_fn(batch):
    imgs, labels, lengths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    labels = torch.cat(labels)
    return imgs, labels, torch.tensor(lengths), torch.tensor([len(imgs)] * len(imgs))

class CRNN(nn.Module):
    def __init__(self, use_lstm=True):
        super().__init__()
        self.use_lstm = use_lstm
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2), nn.BatchNorm2d(512), nn.ReLU(),
        )
        if use_lstm:
            self.lstm = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True)
            self.fc = nn.Linear(512, NUM_CLASS)
        else:
            self.fc = nn.Linear(512, NUM_CLASS)

    def forward(self, x):
        conv = self.cnn(x)
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        if self.use_lstm:
            rnn_out, _ = self.lstm(conv)
            logits = self.fc(rnn_out)
        else:
            logits = self.fc(conv)
        return logits.permute(1, 0, 2)

def decode(pred_logits, blank=0):
    pred = pred_logits.argmax(dim=-1)
    result = []
    prev = blank
    for idx in pred:
        if idx != blank and idx != prev:
            result.append(IDX2CHAR.get(idx.item(), "?"))
        prev = idx
    return "".join(result)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, labels, lengths, _ in tqdm(loader, desc="  Train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        T = logits.size(0)
        input_lengths = torch.full((imgs.size(0),), T, dtype=torch.long)
        loss = criterion(logits, labels, input_lengths, lengths.to(torch.long))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_correct, total_chars = 0, 0
    for imgs, labels, lengths, _ in tqdm(loader, desc="  Eval "):
        imgs = imgs.to(device)
        logits = model(imgs)
        for i in range(imgs.size(0)):
            pred_str = decode(logits[:, i, :].cpu())
            gt_str = "".join(IDX2CHAR[labels[j].item()] for j in range(sum(lengths[:i].tolist()), sum(lengths[:i+1].tolist())))
            total_correct += sum(1 for a, b in zip(pred_str, gt_str) if a == b)
            total_chars += len(gt_str)
    return total_correct / total_chars if total_chars > 0 else 0

def main(use_lstm=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_samples = load_ocr_data("data/ocr/train")
    test_samples  = load_ocr_data("data/ocr/test")
    train_ds, test_ds = OCRDataset(train_samples), OCRDataset(test_samples)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    model = CRNN(use_lstm=use_lstm).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    mode_name = "CRNN (CNN + LSTM)" if use_lstm else "CNN Only (no LSTM)"
    print(f"\n{'='*50}\nTraining: {mode_name}\n{'='*50}")
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:2d} | Loss: {loss:.4f} | Char Acc: {acc*100:.1f}%")
    suffix = "with_lstm" if use_lstm else "no_lstm"
    torch.save(model.state_dict(), f"stage2_crnn_{suffix}.pth")
    print(f"Saved: stage2_crnn_{suffix}.pth")
    return model

@torch.no_grad()
def infer_image(model_path, image_path, use_lstm=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(use_lstm=use_lstm).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    transform = transforms.Compose([transforms.Resize((IMG_H, IMG_W)), transforms.Grayscale(), transforms.ToTensor()])
    img = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    logits = model(img)
    print(f"Result: {decode(logits[:, 0, :].cpu())}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        infer_image("stage2_crnn_with_lstm.pth", sys.argv[2], use_lstm=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "no_lstm":
        main(use_lstm=False)
    else:
        main(use_lstm=True)
```

### 运行命令

```bash
python stage2_crnn_ocr.py                       # train CRNN (with LSTM)
python stage2_crnn_ocr.py no_lstm               # train CNN-only (comparison)
python stage2_crnn_ocr.py test data/ocr/test/hello_1.jpg  # inference
```

### 对比实验：关键差异

| 维度 | CRNN (with LSTM) | CNN Only (no LSTM) |
|------|------------------|--------------------|
| Char Accuracy | 80%+ (20 epochs) | < 40% (scrambled) |
| Output Order | "hello" (correct) | "oellh" / "lhloe" |
| Long Sequences | Stable | Severely degrades |

---

# 第3站：自注意力 —— TrOCR (Transformer OCR)

## 3.1 学习目标

- Self-Attention vs Cross-Attention
- Runtime inference with Microsoft TrOCR
- **Visualize Attention Maps**: see where the model "looks"

## 3.2 核心代码: TrOCR + Attention Visualization (`stage3_trocr_attention.py`)

```python
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def ocr_with_attention(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        output = model.generate(
            pixel_values,
            output_attentions=True,
            return_dict_in_generate=True,
            max_length=128,
        )

    generated_ids = output.sequences[0]
    text = processor.decode(generated_ids, skip_special_tokens=True)
    tokens = processor.tokenizer.convert_ids_to_tokens(generated_ids)

    print(f"\nOCR Result: {text}")
    print(f"Tokens: {tokens}")

    cross_attns = torch.stack(output.cross_attentions, dim=0)
    attn = cross_attns[:, -1, 0]
    attn = attn.mean(dim=1)

    return text, tokens, attn.cpu().numpy(), image

def plot_attention_overlay(image, tokens, attn, save_path="stage3_attention.png"):
    n_tokens = min(len(tokens), 12)
    cols = 4
    rows = (n_tokens + cols - 1) // cols

    img_w, img_h = image.size
    n_patches = int(np.sqrt(attn.shape[1]))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for i in range(n_tokens):
        ax = axes[i]
        ax.imshow(image)
        attn_map = attn[i].reshape(n_patches, n_patches)
        attn_map = np.array(Image.fromarray(attn_map).resize((img_w, img_h), Image.BILINEAR))
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        ax.imshow(attn_map, cmap="jet", alpha=0.5)
        ax.set_title(f'"{tokens[i]}"', fontsize=12)
        ax.axis("off")

    for j in range(n_tokens, len(axes)):
        axes[j].axis("off")

    plt.suptitle("TrOCR Cross-Attention: where the model looks for each token", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Attention map saved: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stage3_trocr_attention.py path/to/image.jpg")
        sys.exit(1)
    text, tokens, attn, image = ocr_with_attention(sys.argv[1])
    plot_attention_overlay(image, tokens, attn)
```

### 运行命令

```bash
python stage3_trocr_attention.py data/ocr/test/world_0.jpg
```

### Understanding the Attention Map

```
OCR Pipeline:
  Input Image (cat.jpg)
      |
  ViT Encoder (splits image into 24x24 patches, 16x16 px each)
      |
  Encoder Output (576 patch feature vectors)
      |
  Decoder generates tokens one by one ("c", "a", "t")
      |  Each token generation uses Cross-Attention over all 576 patches
      v
  Final Output: "cat"

The Attention heatmap shows:
  When generating token "c" -> bright spot on letter "c" position
  When generating token "a" -> bright spot on letter "a" position
  When generating token "t" -> bright spot on letter "t" position
```

---

# 三站对比总结

| Dimension | Stage 1: CNN | Stage 2: CRNN | Stage 3: TrOCR |
|-----------|-------------|---------------|----------------|
| Model Size | Tiny (~0.1M) | Medium (~5M) | Large (~300M+) |
| Feature Method | Conv sliding windows | CNN+LSTM sequence | Self/Cross-Attention |
| Core Insight | Layer depth = detail->semantic | No sequence = scrambled output | Attention = "where it looks" |
| Visualization | Feature Map grid | With/without LSTM comparison | Attention Heatmap overlay |
| Real-world Use | Basic classification | Standard OCR baseline | SOTA OCR / Doc Understanding |

---

# 常见坑点速查

| Issue | Cause | Fix |
|-------|-------|-----|
| CTC Loss not decreasing | input_lengths mismatch | Print logits.size(0), verify T = W // stride |
| LSTM outputs all same | batch_first dimension wrong | Print rnn_out.shape, should be (B, T, hidden*2) |
| TrOCR OOM | Model too large | torch.cuda.empty_cache() or use trocr-small-printed |
| Attention Map uniform | Using Self-Attn not Cross-Attn | Confirm output.cross_attentions |
| trdg images low accuracy | Too much randomness | Reduce -tc color range or -bl blur |

---

# 进阶路线

1. Fine-tune TrOCR on domain data (invoices, medical records)
2. Explore ViT+OCR variants: Donut, Pix2Struct
3. Production deployment: ONNX Runtime / TensorRT quantization
4. Detection + OCR pipeline: YOLO/DBnet for text detection, CRNN/TrOCR for recognition
