# ==============================================================================
# 一、库导入与环境配置
# ==============================================================================
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import T5Tokenizer, T5EncoderModel, AutoConfig
import joblib
import re
from collections import OrderedDict
from sklearn.metrics import (
    confusion_matrix, matthews_corrcoef, roc_auc_score,
    precision_score, recall_score, accuracy_score
)

# 抑制警告
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

MY_SEED = 42

# ======================== 路径与超参数配置 ========================
# 数据路径
PATH_4MER = '/root/autodl-tmp/NPZ/4mer_fasttext.npz'         
PATH_Y_TEST = '/root/autodl-tmp/y_test.csv'
PATH_TEST_SEQ = "/root/autodl-tmp/test_fixed_sequences.csv"

# 模型与权重路径
ANKH_PATH = '/root/autodl-tmp/Big_Model/Ankh3-XL'
ANKH3_EMBED_PREFIX = '[NLU]'
OUTPUT_CSV_PATH = '/root/autodl-tmp/Deep-dsRNAPred_test_probabilities.csv'

# 模型结构超参数 (必须与训练时保持完全一致)
BATCH_SIZE = 128
NUM_CLASSES = 2
MAX_SEQ_LENGTH = 100
PCA_TARGET_DIM = 512
DROPOUT = 0.1

INPUT_SHAPE_4MER = (97, 150)
BRANCH2_INPUT_DIM = PCA_TARGET_DIM
ADAPTIVE_POOL_SIZE_PER_LAYER = [50, 25, 10, 5]

FILTERS_4MER = [256, 512, 256, 256]
SK_KERNELS_4MER = [[1, 3, 5], [1, 3, 5], [1, 3], [1, 3]]    

FILTERS_ANKH = [512, 1024, 512, 512]
SK_KERNELS_ANKH = [[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]]

# ==============================================================================
# 二、工具函数 
# ==============================================================================
def clean_ankh_sequence(seq):
    return re.sub(r"[UZOB]", "X", str(seq).strip().upper())

def assert_ankh_token_alignment(clean_seq, tokenizer, prefix_token_ids, input_ids_row, attention_mask_row, sequence_index):
    seq_ids = tokenizer.encode(clean_seq, add_special_tokens=False)
    combined_ids = tokenizer.encode(f"{ANKH3_EMBED_PREFIX}{clean_seq}", add_special_tokens=False)
    prefix_token_count = len(prefix_token_ids)
    seq_end = prefix_token_count + len(seq_ids)
    return len(seq_ids)

def get_valid_token_lengths(sequences, max_seq_length):
    return np.array([min(len(clean_ankh_sequence(seq)), max_seq_length) for seq in sequences], dtype=np.int32)

def transform_valid_token_rows(data_3d, lengths, max_seq_length, scaler, pca):
    num_samples = data_3d.shape[0]
    max_available_len = data_3d.shape[1]
    output_dim = int(getattr(pca, 'n_components_', getattr(pca, 'n_components')))
    out = np.zeros((num_samples, max_seq_length, output_dim), dtype=np.float32)
    for sample_idx, valid_len in enumerate(lengths):
        valid_len = int(max(0, min(valid_len, max_seq_length, max_available_len)))
        if valid_len == 0: continue
        sample_valid_rows = data_3d[sample_idx, :valid_len, :]
        sample_valid_rows = scaler.transform(sample_valid_rows)
        sample_valid_rows = pca.transform(sample_valid_rows).astype(np.float32)
        out[sample_idx, :valid_len, :] = sample_valid_rows
    return out

def apply_scaler_pca_test(val_3d, max_seq_length, val_lengths, scaler, pca):
    out_val = transform_valid_token_rows(val_3d, val_lengths, max_seq_length, scaler, pca)
    return out_val

def extract_ankh_features(sequences, max_seq_length, tokenizer, ankh_model, device):
    hidden_states_list = []
    ankh_model.eval()
    prefix_token_ids = tokenizer.encode(ANKH3_EMBED_PREFIX, add_special_tokens=False)
    prefix_token_count = len(prefix_token_ids)
    
    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch_seq = sequences[i:i + BATCH_SIZE].tolist()
            clean_batch_seq = [clean_ankh_sequence(seq) for seq in batch_seq]
            sequence_examples = [f"{ANKH3_EMBED_PREFIX}{seq}" for seq in clean_batch_seq]
            ids = tokenizer.batch_encode_plus(
                sequence_examples, add_special_tokens=True, is_split_into_words=False,
                padding="longest", return_tensors="pt"
            )
            input_ids = ids['input_ids'].to(device)
            attention_mask = ids['attention_mask'].to(device)
            outputs = ankh_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            batch_hidden = []
            for idx, clean_seq in enumerate(clean_batch_seq):
                residue_token_count = assert_ankh_token_alignment(
                    clean_seq, tokenizer, prefix_token_ids,
                    ids['input_ids'][idx], ids['attention_mask'][idx], i + idx
                )
                seq_start = prefix_token_count
                seq_end = seq_start + residue_token_count
                seq_emb = last_hidden_state[idx, seq_start:seq_end]
                
                if seq_emb.shape[0] > max_seq_length:
                    seq_emb = seq_emb[:max_seq_length, :]
                elif seq_emb.shape[0] < max_seq_length:
                    pad_len = max_seq_length - seq_emb.shape[0]
                    seq_emb = F.pad(seq_emb, (0, 0, 0, pad_len))
                batch_hidden.append(seq_emb.unsqueeze(0))
            batch_hidden = torch.cat(batch_hidden, dim=0)
            hidden_states_list.append(batch_hidden.cpu())
    all_hidden = torch.cat(hidden_states_list, dim=0)
    return all_hidden.numpy()

def read_csv_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    df = df[['sequence', 'label']].copy()
    df['sequence'] = df['sequence'].str.strip().str.upper()
    df['label'] = df['label'].astype(int)
    df = df.reset_index(drop=True)
    return df

def calculate_metrics_original(y_true, y_pred_prob):
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred_prob, axis=1)
    y_pred_pos_prob = y_pred_prob[:, 1]
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    if cm.shape == (1, 1):
        tn, fp, fn, tp = (cm[0, 0], 0, 0, 0) if y_true_labels[0] == 0 else (0, 0, 0, cm[0, 0])
    else:
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    mcc = matthews_corrcoef(y_true_labels, y_pred_labels)
    auc_score = roc_auc_score(y_true_labels, y_pred_pos_prob) if len(np.unique(y_true_labels)) > 1 else 0.5
    precision = precision_score(y_true_labels, y_pred_labels, zero_division=0)
    recall = recall_score(y_true_labels, y_pred_labels, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    return accuracy, mcc, auc_score, precision, recall, specificity, sensitivity

# ==============================================================================
# 三、模型定义 (完整保留)
# ==============================================================================
class ECA1dAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x).squeeze(-1).unsqueeze(1)
        y = self.conv(y)
        y = self.sigmoid(y).permute(0, 2, 1)
        return x * y

class Conv1dLayerNormBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dropout=0.):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return x

class SKAttention1D(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            if k % 2 == 1:
                pad_layer = nn.Identity()
                conv_padding = (k - 1) // 2
            else:
                pad_left = k // 2
                pad_right = k // 2 - 1
                pad_layer = nn.ConstantPad1d((pad_left, pad_right), 0)
                conv_padding = 0
            self.convs.append(nn.Sequential(OrderedDict([
                ('pad', pad_layer),
                ('conv', nn.Conv1d(channel, channel, kernel_size=k, padding=conv_padding, groups=group)),
                ('block', Conv1dLayerNormBlock(channel, channel, kernel_size=1, padding=0)),
                ('relu', nn.ReLU(inplace=True))
            ])))
        self.fc = nn.Sequential(
            nn.Linear(channel, self.d, bias=False),
            nn.BatchNorm1d(self.d),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([nn.Linear(self.d, channel) for _ in kernels])
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, seq_len = x.size()
        conv_outs = [conv(x) for conv in self.convs]
        feats = torch.stack(conv_outs, 0)
        U = sum(conv_outs)
        S = U.mean(-1)
        Z = self.fc(S)
        weights = [fc(Z).view(bs, c, 1) for fc in self.fcs]
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        V = (attention_weights * feats).sum(0)
        return V

class SKResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernels, eca_kernel_size=3,
                 reduction=16, group=1, L=32, adaptive_pool_size=25, dropout=0.2):
        super().__init__()
        self.sk_attention = SKAttention1D(channel=in_channels, kernels=kernels,
                                          reduction=reduction, group=group, L=L)
        self.eca_attention = ECA1dAttention(kernel_size=eca_kernel_size)
        self.channel_map = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.dropout_layer = nn.Dropout(dropout)
        self.pool = nn.AdaptiveMaxPool1d(output_size=adaptive_pool_size)
        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual.add_module('res_conv', nn.Conv1d(in_channels, out_channels, 1))
        self.residual.add_module('res_pool', nn.AdaptiveMaxPool1d(output_size=adaptive_pool_size))

    def forward(self, x):
        residual = x
        out = self.sk_attention(x)
        out = self.eca_attention(out)
        out = self.channel_map(out)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.pool(out)
        residual = self.residual(residual)
        out = out + residual
        out = self.relu(out)
        return out

class ModelConfig:
    def __init__(self, device, max_time_steps, input_size, num_classes, filters=None, sk_kernels=None):
        self.device = device
        self.max_time_steps = max_time_steps
        self.input_size = input_size
        self.num_classes = num_classes
        self.dropout = DROPOUT
        self.num_cnn_layers = 4
        self.sk_kernels_per_layer = sk_kernels if sk_kernels is not None else [[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.sk_reduction = 16
        self.sk_group = 1
        self.sk_L = 32
        self.filters = filters if filters is not None else [1024, 2048, 1024, 1024]
        self.adaptive_pool_sizes = ADAPTIVE_POOL_SIZE_PER_LAYER
        self.eca_kernel_size = 3

class ProtBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        in_channels = config.input_size
        for i in range(config.num_cnn_layers):
            self.conv_blocks.append(
                SKResidualBlock1D(
                    in_channels=in_channels,
                    out_channels=config.filters[i],
                    kernels=config.sk_kernels_per_layer[i],
                    eca_kernel_size=config.eca_kernel_size,
                    reduction=config.sk_reduction,
                    group=config.sk_group,
                    L=config.sk_L,
                    adaptive_pool_size=config.adaptive_pool_sizes[i],
                    dropout=config.dropout
                )
            )
            in_channels = config.filters[i]
        self.final_dim = config.filters[-1] * config.adaptive_pool_sizes[-1]
        self.fc_block = nn.Sequential(
            nn.Linear(self.final_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for block in self.conv_blocks:
            x = block(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_block(x)
        return x

class TwoBranchCombinedModel(nn.Module):
    def __init__(self, model_4mer, model_prot):
        super().__init__()
        self.model_4mer = model_4mer
        self.model_prot = model_prot
        self.branch_out_dim = 32
        self.combined_layers = nn.Sequential(
            nn.Linear(self.branch_out_dim * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x4, xprot):
        out4 = self.model_4mer(x4)
        outprot = self.model_prot(xprot)
        combined = torch.cat([out4, outprot], dim=-1)
        out = self.combined_layers(combined)
        return out


# ==============================================================================
# 四、独立测试与概率导出主函数
# ==============================================================================
def run_test_and_save_probs():
    print("===================== Test Set Inference & Export =====================")
    
    # 1. 加载测试数据
    print("1. 加载 4-mer 特征与标签...")
    data_4mer = np.load(PATH_4MER)
    x_test_4mer = data_4mer['x_test'].astype(np.float32)
    
    y_test_df = pd.read_csv(PATH_Y_TEST)
    y_test_onehot = np.eye(NUM_CLASSES)[y_test_df.to_numpy().astype(np.int32).reshape(-1)]
    y_test_label = np.argmax(y_test_onehot, axis=1)

    print("2. 加载序列并提取 Ankh 特征...")
    test_seq_df = read_csv_data(PATH_TEST_SEQ)
    test_seqs = test_seq_df['sequence'].values
    test_token_lengths = get_valid_token_lengths(test_seqs, MAX_SEQ_LENGTH)

    tokenizer = T5Tokenizer.from_pretrained(ANKH_PATH)
    ankh_config = AutoConfig.from_pretrained(f"{ANKH_PATH}/config.json")
    ankh_model = T5EncoderModel.from_pretrained(ANKH_PATH, config=ankh_config).to(DEVICE)
    X_test_plm = extract_ankh_features(test_seqs, MAX_SEQ_LENGTH, tokenizer, ankh_model, DEVICE)
    
    # 释放显存
    del ankh_model
    torch.cuda.empty_cache()

    # 2. 配置模型架构
    config_4mer = ModelConfig(DEVICE, INPUT_SHAPE_4MER[0], INPUT_SHAPE_4MER[1], NUM_CLASSES,
                              filters=FILTERS_4MER, sk_kernels=SK_KERNELS_4MER)
    config_prot = ModelConfig(DEVICE, MAX_SEQ_LENGTH, BRANCH2_INPUT_DIM, NUM_CLASSES,
                              filters=FILTERS_ANKH, sk_kernels=SK_KERNELS_ANKH)

    # 3. 循环加载 5 折模型进行软投票 (Soft Voting)
    all_fold_probs = []
    
    for fold_idx in range(1, 6):
        f_model_path = f"best_model_fold_{fold_idx}.pth"
        f_scaler_path = f"best_scaler_fold_{fold_idx}.pkl"
        f_pca_path = f"best_pca_fold_{fold_idx}.pkl"

        if not (os.path.exists(f_model_path) and os.path.exists(f_scaler_path)):
            print(f"警告: Fold {fold_idx} 权重或字典缺失，跳过。")
            continue

        print(f"正在加载 Fold {fold_idx} 权重进行推理...")

        # 降维处理
        f_scaler = joblib.load(f_scaler_path)
        f_pca = joblib.load(f_pca_path)
        X_test_prot_fold = apply_scaler_pca_test(
            val_3d=X_test_plm, max_seq_length=MAX_SEQ_LENGTH,
            val_lengths=test_token_lengths, scaler=f_scaler, pca=f_pca
        )

        # 构建 DataLoader
        test_dataset = TensorDataset(
            torch.from_numpy(x_test_4mer),
            torch.from_numpy(X_test_prot_fold),
            torch.from_numpy(y_test_label).long()
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 初始化并加载模型
        model_4mer = ProtBertModel(config_4mer).to(DEVICE)
        model_prot = ProtBertModel(config_prot).to(DEVICE)
        model_fold = TwoBranchCombinedModel(model_4mer, model_prot).to(DEVICE)
        model_fold.load_state_dict(torch.load(f_model_path, map_location=DEVICE))
        model_fold.eval()

        # 推理获取输出
        fold_preds = []
        with torch.no_grad():
            for batch in test_loader:
                x4, xprot, _ = batch
                x4, xprot = x4.to(DEVICE), xprot.to(DEVICE)
                outputs = model_fold(x4, xprot)
                fold_preds.append(outputs.cpu().numpy())

        fold_preds = np.concatenate(fold_preds, axis=0)
        # 转换为 Softmax 概率
        fold_probs = torch.softmax(torch.from_numpy(fold_preds), dim=-1).numpy()
        all_fold_probs.append(fold_probs)

    if not all_fold_probs:
        print("集成失败：未能加载任何模型的验证折。")
        return

    # 4. 软投票 (计算各折预测概率的平均值)
    print("\n执行软投票 (Soft Voting) ...")
    ensemble_probs = np.mean(all_fold_probs, axis=0)
    
    # 计算评估指标
    test_metrics = calculate_metrics_original(y_test_onehot, ensemble_probs)
    test_acc, test_mcc, test_auc, test_prec, test_rec, test_spec, test_sens = test_metrics

    print(f"\n[最终集成评估结果]")
    print(f"Test Accuracy: {test_acc:.4f}, Test AUC: {test_auc:.4f}")
    print(f"Test MCC: {test_mcc:.4f}, Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}, Test Specificity: {test_spec:.4f}, Test Sensitivity: {test_sens:.4f}")

    # 5. 保存概率至表格
    print("\n保存预测概率...")
    
    # 提取正类(Class 1)和负类(Class 0)的预测概率
    prob_class_0 = ensemble_probs[:, 0]
    prob_class_1 = ensemble_probs[:, 1]
    
    # 结合真实标签和序列生成 DataFrame
    results_df = pd.DataFrame({
        'Sequence': test_seqs,
        'True_Label': y_test_label,
        'Prob_Class_0': prob_class_0,
        'Prob_Class_1': prob_class_1,
        'Predicted_Label': np.argmax(ensemble_probs, axis=1)
    })
    
    results_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"测试集概率及预测结果已成功导出至: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    run_test_and_save_probs()