import os
os.environ['PADDLE_NLP_DISABLE_AISTUDIO'] = '1'

import argparse

import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader
from tqdm import tqdm

from paddlenlp.transformers import AutoTokenizer, AutoModel


# ==================== Model Definition ====================
class CPAModel(nn.Layer):
    """
    表格语义关系提取模型 (CPA - Column Pair Analysis)
    
    架构设计：
    1. 预训练编码器：BERT/RoBERTa
    2. 特征聚合：支持多种 Pooling 策略
    3. 多头注意力融合：增强特征表达能力
    """
    def __init__(self, model_name, num_labels, dropout_rate=0.1, pooling_strategy='cls_mean_max', use_multi_head=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.pooling_strategy = pooling_strategy
        self.use_multi_head = use_multi_head

        # 自动推断隐藏层维度
        hidden_size = self._infer_hidden_size()
        
        # 根据 pooling 策略确定分类器输入维度
        classifier_input_dim = self._calculate_input_dim(hidden_size)

        # 多头注意力特征融合
        if use_multi_head:
            self.multi_head_fusion = nn.MultiHeadAttention(
                embed_dim=classifier_input_dim, 
                num_heads=min(8, classifier_input_dim // 64)
            )
            self.layer_norm = nn.LayerNorm(classifier_input_dim)
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, classifier_input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(classifier_input_dim // 2, num_labels)
            )
        else:
            self.classifier = nn.Linear(classifier_input_dim, num_labels)

    def _infer_hidden_size(self):
        """自动推断预训练模型的隐藏层维度"""
        hidden_size = None
        if hasattr(self.encoder, 'config'):
            if isinstance(self.encoder.config, dict):
                hidden_size = self.encoder.config.get('hidden_size', None)
            else:
                hidden_size = getattr(self.encoder.config, 'hidden_size', None)

        if hidden_size is None and hasattr(self.encoder, 'embeddings'):
            if hasattr(self.encoder.embeddings, 'word_embeddings'):
                hidden_size = self.encoder.embeddings.word_embeddings.weight.shape[-1]

        if hidden_size is None:
            raise ValueError('无法自动推断 hidden_size，请检查模型配置')
        
        return hidden_size

    def _calculate_input_dim(self, hidden_size):
        """根据 pooling 策略计算分类器输入维度"""
        if self.pooling_strategy == 'cls_mean_max':
            return hidden_size * 3
        elif self.pooling_strategy in ['cls_mean', 'cls_max']:
            return hidden_size * 2
        elif self.pooling_strategy == 'triple':
            return hidden_size * 3
        else:
            return hidden_size

    def _pooling(self, sequence_output, attention_mask, cls_embedding):
        """根据策略进行特征聚合"""
        if self.pooling_strategy == 'cls':
            return cls_embedding
        elif self.pooling_strategy == 'mean':
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).astype('float32')
            sum_embeddings = paddle.sum(sequence_output * mask_expanded, axis=1)
            sum_mask = paddle.clamp(mask_expanded.sum(axis=1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling_strategy == 'max':
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).astype('float32')
            sequence_output = sequence_output.masked_fill(mask_expanded == 0, -1e9)
            return paddle.max(sequence_output, axis=1)
        elif self.pooling_strategy == 'cls_mean':
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).astype('float32')
            sum_embeddings = paddle.sum(sequence_output * mask_expanded, axis=1)
            sum_mask = paddle.clamp(mask_expanded.sum(axis=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            return paddle.concat([cls_embedding, mean_pooled], axis=1)
        elif self.pooling_strategy == 'cls_max':
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).astype('float32')
            masked_output = sequence_output.masked_fill(mask_expanded == 0, -1e9)
            max_pooled = paddle.max(masked_output, axis=1)
            return paddle.concat([cls_embedding, max_pooled], axis=1)
        elif self.pooling_strategy == 'cls_mean_max':
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).astype('float32')
            sum_embeddings = paddle.sum(sequence_output * mask_expanded, axis=1)
            sum_mask = paddle.clamp(mask_expanded.sum(axis=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            masked_output = sequence_output.masked_fill(mask_expanded == 0, -1e9)
            max_pooled = paddle.max(masked_output, axis=1)
            return paddle.concat([cls_embedding, mean_pooled, max_pooled], axis=1)
        else:
            return cls_embedding

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if isinstance(outputs, tuple):
            sequence_output = outputs[0]
        elif hasattr(outputs, 'last_hidden_state'):
            sequence_output = outputs.last_hidden_state
        else:
            sequence_output = outputs

        cls_embedding = sequence_output[:, 0, :]
        pooled = self._pooling(sequence_output, attention_mask, cls_embedding)
        pooled = self.dropout(pooled)
        
        if self.use_multi_head:
            pooled = pooled.unsqueeze(1)
            pooled = self.multi_head_fusion(pooled, pooled, pooled)
            pooled = pooled.squeeze(1)
            pooled = self.layer_norm(pooled)
        
        logits = self.classifier(pooled)
        return logits


# ==================== Tokenization Helper ====================
def encode_text(tokenizer, text_input, max_length):
    """
    将文本编码为模型输入格式
    :param tokenizer: 预训练 tokenizer
    :param text_input: 输入文本
    :param max_length: 最大序列长度
    :return: input_ids, attention_mask
    """
    try:
        encoding = tokenizer(
            text_input,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
        )
    except TypeError:
        encoding = tokenizer(
            text=text_input,
            max_seq_len=max_length,
            pad_to_max_seq_len=True,
            truncation=True,
            return_attention_mask=True,
        )

    input_ids = encoding['input_ids']
    attention_mask = encoding.get('attention_mask', None)

    if attention_mask is None:
        seq_len = encoding.get('seq_len', len(input_ids))
        seq_len = min(seq_len, max_length)
        attention_mask = [1] * seq_len + [0] * (max_length - seq_len)

    return np.array(input_ids, dtype='int64'), np.array(attention_mask, dtype='int64')


# ==================== Inference Dataset ====================
class InferenceDataset(Dataset):
    """推理数据集"""
    def __init__(self, csv_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.original_rows = []
        
        df = pd.read_csv(csv_path, low_memory=False, encoding='utf-8-sig')
        df.columns = [str(col).strip() for col in df.columns]
        
        subject_col = None
        object_col = None
        for col in df.columns:
            if col.lower() == 'subject':
                subject_col = col
            elif col.lower() == 'object':
                object_col = col
        
        if subject_col is None or object_col is None:
            raise ValueError("CSV 文件必须包含 'Subject' 和 'Object' 列")
        
        temp_df = df[[subject_col, object_col]].dropna()
        for idx, row in temp_df.iterrows():
            self.samples.append((str(row[subject_col]), str(row[object_col])))
            self.original_rows.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        subject_text, object_text = self.samples[idx]
        text_input = f"{subject_text} [SEP] {object_text}"
        input_ids, attention_mask = encode_text(self.tokenizer, text_input, self.max_length)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'orig_idx': np.int64(idx),
        }


def collate_fn(samples):
    """批量处理函数"""
    return {
        'input_ids': np.stack([s['input_ids'] for s in samples]).astype('int64'),
        'attention_mask': np.stack([s['attention_mask'] for s in samples]).astype('int64'),
        'orig_idx': np.array([s['orig_idx'] for s in samples], dtype='int64'),
    }


# ==================== Device Helper ====================
def resolve_device(device_arg):
    """解析并设置计算设备"""
    try:
        custom_types = paddle.device.get_all_custom_device_type()
    except Exception:
        custom_types = []

    print(f'可用自定义设备类型: {custom_types}')

    if device_arg and device_arg != 'auto':
        try:
            dev = paddle.set_device(device_arg)
            print(f'使用指定设备: {dev}')
            return dev
        except Exception as e:
            print(f'{device_arg} 设备使用失败: {e}')
    
    # 自动检测可用设备
    if 'iluvatar_gpu' in custom_types:
        try:
            dev = paddle.set_device('iluvatar_gpu')
            print(f'自动检测并使用 Iluvatar GPU: {dev}')
            return dev
        except Exception:
            pass
    
    if 'xpu' in custom_types:
        try:
            dev = paddle.set_device('xpu')
            print(f'自动检测并使用 XPU: {dev}')
            return dev
        except Exception:
            pass
    
    if 'gpu' in custom_types:
        try:
            dev = paddle.set_device('gpu')
            print(f'自动检测并使用 GPU: {dev}')
            return dev
        except Exception:
            pass

    dev = paddle.set_device('cpu')
    print('回退到 CPU.')
    return dev


# ==================== Inference Pipeline ====================
def run_inference(args):
    """主推理函数"""
    device = resolve_device(args.device)
    
    # 加载标签映射
    print(f"加载标签文件: {args.labels_path}")
    with open(args.labels_path, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]
    id2label = {idx: label for idx, label in enumerate(classes)}
    num_labels = len(classes)
    print(f"标签数量: {num_labels}")
    
    # 初始化 tokenizer 和模型
    print(f"加载 Tokenizer: {args.shortcut_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    
    print(f"加载模型: {args.model_path}")
    model = CPAModel(
        args.shortcut_name, 
        num_labels, 
        pooling_strategy=args.pooling_strategy, 
        use_multi_head=args.use_multi_head
    )
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'模型文件不存在: {args.model_path}')
    
    state_dict = paddle.load(args.model_path)
    model.set_state_dict(state_dict)
    model.eval()
    print("模型加载完成")
    
    # 加载数据集
    print(f"加载测试数据集: {args.input_csv}")
    dataset = InferenceDataset(args.input_csv, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        return_list=True,
    )
    
    print(f"开始推理. 有效行数: {len(dataset)}")
    predictions = [None] * len(dataset)
    use_amp = args.use_amp and str(device) != 'cpu'
    
    with paddle.no_grad():
        for batch in tqdm(dataloader, desc='推理进度'):
            input_ids = paddle.to_tensor(batch['input_ids'], dtype='int64')
            attention_mask = paddle.to_tensor(batch['attention_mask'], dtype='int64')
            orig_indices = batch['orig_idx'].tolist()
            
            if use_amp:
                with paddle.amp.auto_cast(enable=True):
                    logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)
            
            preds = paddle.argmax(logits, axis=1).numpy().tolist()
            for idx_in_batch, pred_idx in enumerate(preds):
                original_position = orig_indices[idx_in_batch]
                predictions[original_position] = id2label[pred_idx]
    
    # 保存结果
    original_df = pd.read_csv(args.input_csv, low_memory=False, encoding='utf-8-sig')
    original_df.columns = [str(col).strip() for col in original_df.columns]
    
    subject_col = None
    object_col = None
    for col in original_df.columns:
        if col.lower() == 'subject':
            subject_col = col
        elif col.lower() == 'object':
            object_col = col
    
    if subject_col is None or object_col is None:
        raise ValueError("CSV 文件必须包含 'Subject' 和 'Object' 列")
    
    valid_mask = original_df[subject_col].notna() & original_df[object_col].notna()
    valid_indices = original_df[valid_mask].index.tolist()
    
    original_df['Label'] = None
    for row_idx, pred_label in zip(valid_indices, predictions):
        original_df.loc[row_idx, 'Label'] = pred_label
    
    original_df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
    print(f"推理完成! 结果已保存到: {args.output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='表格语义关系提取模型推理 (CPA - Column Pair Analysis)')
    
    parser.add_argument('--input_csv', type=str, default="./dataset/test.csv",
                        help='测试集 CSV 文件路径')
    parser.add_argument('--labels_path', type=str, required=True,
                        help='标签类别文件路径')
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型权重路径')
    parser.add_argument('--output_file', type=str, default='./submission.csv',
                        help='预测结果输出路径')
    parser.add_argument('--shortcut_name', type=str, default='bert-large-uncased',
                        help='预训练模型名称（需与训练一致）')
    parser.add_argument('--pooling_strategy', type=str, default='cls_mean_max',
                        choices=['cls', 'mean', 'max', 'cls_mean', 'cls_max', 'cls_mean_max'],
                        help='特征聚合策略（需与训练一致）')
    parser.add_argument('--use_multi_head', action='store_true', default=False,
                        help='使用多头注意力（需与训练一致）')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批处理大小')
    parser.add_argument('--max_length', type=int, default=128,
                        help='最大序列长度（需与训练一致）')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载线程数')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备类型: auto, cpu, gpu, xpu')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='启用自动混合精度推理')
    
    args = parser.parse_args()
    run_inference(args)
