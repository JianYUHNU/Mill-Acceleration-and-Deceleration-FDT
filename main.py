"""
main.py
FDT 出口温度回归
1. 自动剔除无关列（手动 + 统计规则）
2. 训练 / 验证 / TensorBoard / torchinfo
3. 权重自动加载或重新训练
"""
import os, glob, datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
from torchinfo import summary

# --------------------------------------------------
# 1. 路径
# --------------------------------------------------
ROOT_DIR    = r"C:\Users\简语\Desktop\轧机升速降速FDT"
SAVE_DIR    = "runs"
WEIGHT_PATH = "best_model.pt"
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------
# 2. 手动剔除列表
# --------------------------------------------------
MANUAL_DROP = [
    "Date", "Time", "Milli Sec",
    "L_FMTSTODG_10", "L_FM_F7SFBMS", "L_FM_F6MSUC",
    "L_FM_MRH_MANACC", "L_FM_MRH_MANDEC",
    "L_MPD069PL_SL", "L_MPD070PL_SL", "L_MPD071PL_SL",
    "L_SUC_F5TSUC", "L_SUC_F6TSUC"
]
TARGET_COL  = "L_YP09BT11_IF"

# --------------------------------------------------
# 3. 读取 & 合并 CSV
# --------------------------------------------------
all_csv = [sorted(glob.glob(os.path.join(ROOT_DIR, d, "*.csv")))[0]
           for d in os.listdir(ROOT_DIR)
           if os.path.isdir(os.path.join(ROOT_DIR, d))]
df_list = [pd.read_csv(f).select_dtypes(include=[np.number]) for f in tqdm(all_csv, desc="读取 CSV")]
df = pd.concat(df_list, ignore_index=True)

# --------------------------------------------------
# 4. 特征自动选择（已修复布尔索引长度问题）
# --------------------------------------------------
def auto_select(df, target_col, manual_drop, var_thresh=1e-4, corr_thresh=0.01):
    """返回 (clean_df, kept_columns)"""
    # 1. 手动剔除
    df = df.drop(columns=manual_drop, errors='ignore')

    # 2. 缺失率 >1% 剔除
    df = df.loc[:, df.isnull().mean() < 0.01]

    # 3. 方差阈值（仅对特征）
    X_df = df.drop(columns=[target_col])
    selector = VarianceThreshold(threshold=var_thresh)
    selector.fit(X_df)
    X_df = X_df.loc[:, selector.get_support()]

    # 4. 与因变量相关性阈值
    corr = X_df.corrwith(df[target_col]).abs()
    keep_mask = corr > corr_thresh
    X_df = X_df.loc[:, keep_mask]

    # 5. 加回目标列
    clean_df = pd.concat([X_df, df[target_col]], axis=1)
    return clean_df, X_df.columns.tolist()

df_clean, kept_features = auto_select(df, TARGET_COL, MANUAL_DROP)
print(f"原始特征数：{df.shape[1]-1} → 保留特征数：{len(kept_features)}")
pd.Series(kept_features).to_csv("selected_features.csv", index=False, header=False)

# --------------------------------------------------
# 5. PyTorch 数据准备
# --------------------------------------------------
X = df_clean.drop(columns=[TARGET_COL]).values.astype(np.float32)
y = df_clean[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

# --------------------------------------------------
# 6. 模型
# --------------------------------------------------
class FDTRegressor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FDTRegressor(in_features=X.shape[1]).to(device)

# --------------------------------------------------
# 7. 权重加载 / 训练
# --------------------------------------------------
if os.path.isfile(WEIGHT_PATH):
    print("✅ 检测到权重文件，直接加载...")
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
else:
    print("👉 权重不存在，开始训练...")

    run_tag = datetime.datetime.now().strftime("%m%d_%H%M")
    writer = SummaryWriter(log_dir=os.path.join(SAVE_DIR, f"run_{run_tag}"))
    dummy = torch.randn(1, X.shape[1]).to(device)
    writer.add_graph(model, dummy)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val = float("inf")

    EPOCHS = 100
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item() * xb.size(0)
            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            global_step += 1
            pbar.set_postfix(loss=loss.item())
        train_loss /= len(train_loader.dataset)
        writer.add_scalar("Loss/train_epoch", train_loss, epoch)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch)

        tqdm.write(f"Epoch {epoch:03d} | "
                   f"Train MAE {train_loss:.4f} | Val MAE {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), WEIGHT_PATH)
            tqdm.write("✅ 已保存 best_model.pt")
    writer.close()
    print("训练完成！")

# --------------------------------------------------
# 8. torchinfo 打印参数
# --------------------------------------------------
print("\n========== Model Summary ==========")
summary(model, input_size=(1, X.shape[1]), device=str(device))

# --------------------------------------------------
# 9. 提示
# --------------------------------------------------
print("\n若想查看可视化，请在终端执行：")
print(f"tensorboard --logdir {os.path.abspath(SAVE_DIR)}")