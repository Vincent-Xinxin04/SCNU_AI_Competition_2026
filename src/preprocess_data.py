import argparse
import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import AutoTokenizer


def load_data_from_directory(dir_path):
    all_data = []
    if not os.path.exists(dir_path):
        raise ValueError(f"can't find: {dir_path}")

    try:
        csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    except PermissionError:
        logging.error(f"Permission denied: {dir_path}")
        return []
    logging.info(f"load data from {dir_path} ...")

    for filename in tqdm(csv_files, desc=f"loading {os.path.basename(dir_path)}"):
        file_path = os.path.join(dir_path, filename)
        label_name = filename[:-4]
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding='utf-8-sig')
            if df.empty:
                continue
            df.columns = [str(col).strip() for col in df.columns]
            
            subject_col = None
            object_col = None
            for col in df.columns:
                if col.lower() == 'subject':
                    subject_col = col
                elif col.lower() == 'object':
                    object_col = col
            
            if subject_col is not None and object_col is not None:
                df = df[[subject_col, object_col]].dropna()
                df['label'] = label_name
                all_data.append(df)
        except Exception as e:
            logging.warning(f"{filename} load error: {e}")

    if not all_data:
        raise ValueError(f"{dir_path} not valid data")

    full_df = pd.concat(all_data, ignore_index=True)
    
    subject_col = None
    object_col = None
    for col in full_df.columns:
        if col.lower() == 'subject':
            subject_col = col
        elif col.lower() == 'object':
            object_col = col
    
    full_df[subject_col] = full_df[subject_col].astype(str)
    full_df[object_col] = full_df[object_col].astype(str)
    return full_df


def encode_pair(tokenizer, text_a, text_b, max_length):
    encoding = tokenizer(
        text=text_a,
        text_pair=text_b,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors=None
    )
    
    input_ids = encoding['input_ids']
    attention_mask = encoding.get('attention_mask', None)

    if attention_mask is None:
        seq_len = len(input_ids)
        seq_len = min(seq_len, max_length)
        attention_mask = [1] * seq_len + [0] * (max_length - seq_len)

    return np.array(input_ids, dtype='int64'), np.array(attention_mask, dtype='int64')


def preprocess_data(dataframe, tokenizer, label_encoder, max_length):
    subject_col = None
    object_col = None
    for col in dataframe.columns:
        if col.lower() == 'subject':
            subject_col = col
        elif col.lower() == 'object':
            object_col = col
    
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for idx, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc='Preprocessing'):
        subject_text = str(row[subject_col])
        object_text = str(row[object_col])
        input_ids, attention_mask = encode_pair(tokenizer, subject_text, object_text, max_length)
        label_id = label_encoder.transform([row['label']])[0]
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(np.int64(label_id))
    
    return np.array(input_ids_list), np.array(attention_mask_list), np.array(labels_list)


def main():
    parser = argparse.ArgumentParser(description='Preprocess data and generate cache')
    parser.add_argument('--train_dir', type=str, default="./dataset/Train_Set")
    parser.add_argument('--shortcut_name', type=str, default='bert-base-chinese')
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--force_rebuild', action='store_true', help='Force rebuild cache even if it exists')
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
    )
    
    logging.info('=== Start data preprocessing ===')
    logging.info(f'Train directory: {args.train_dir}')
    logging.info(f'Model: {args.shortcut_name}')
    logging.info(f'Max length: {args.max_length}')
    
    # 加载原始数据
    raw_train_df = load_data_from_directory(args.train_dir)
    
    # 创建标签编码器
    label_encoder = LabelEncoder()
    label_encoder.fit(raw_train_df['label'].unique())
    num_classes = len(label_encoder.classes_)
    logging.info(f'Number of labels: {num_classes}')
    
    # 划分训练集和验证集
    counts = raw_train_df['label'].value_counts()
    rare_labels = counts[counts < 2].index
    df_rare = raw_train_df[raw_train_df['label'].isin(rare_labels)]
    df_common = raw_train_df[~raw_train_df['label'].isin(rare_labels)]

    if len(df_common) == 0:
        raise ValueError("data num < 2, can't split dataset")

    train_c, val_c = train_test_split(
        df_common,
        test_size=args.val_ratio,
        stratify=df_common['label'],
        random_state=args.random_seed,
    )
    train_df = pd.concat([train_c, df_rare]).sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    val_df = val_c.reset_index(drop=True)
    logging.info(f'Split success: train={len(train_df)}, val={len(val_df)}')
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.shortcut_name)
    
    # 设置缓存目录
    cache_dir = os.path.join(os.path.dirname(args.train_dir), 'cache')
    cache_suffix = f'{args.shortcut_name.replace("/", "_")}_len{args.max_length}'
    os.makedirs(cache_dir, exist_ok=True)
    
    # 保存标签类
    label_classes_path = os.path.join(cache_dir, f'label_classes_{cache_suffix}.txt')
    with open(label_classes_path, 'w', encoding='utf-8') as f:
        for label in label_encoder.classes_:
            f.write(f'{label}\n')
    logging.info(f'Label classes saved to {label_classes_path}')
    
    # 检查缓存是否存在
    train_cache_path = os.path.join(cache_dir, f'train_{cache_suffix}.npz')
    val_cache_path = os.path.join(cache_dir, f'val_{cache_suffix}.npz')
    
    if not args.force_rebuild and os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        logging.info('Cache already exists! Use --force_rebuild to rebuild.')
        return
    
    # 预处理训练集
    logging.info('Preprocessing training set...')
    train_input_ids, train_attention_mask, train_labels = preprocess_data(
        train_df, tokenizer, label_encoder, args.max_length
    )
    
    # 预处理验证集
    logging.info('Preprocessing validation set...')
    val_input_ids, val_attention_mask, val_labels = preprocess_data(
        val_df, tokenizer, label_encoder, args.max_length
    )
    
    # 保存训练集缓存
    logging.info(f'Saving training cache to {train_cache_path}')
    np.savez_compressed(
        train_cache_path,
        input_ids=train_input_ids,
        attention_mask=train_attention_mask,
        labels=train_labels
    )
    
    # 保存验证集缓存
    logging.info(f'Saving validation cache to {val_cache_path}')
    np.savez_compressed(
        val_cache_path,
        input_ids=val_input_ids,
        attention_mask=val_attention_mask,
        labels=val_labels
    )
    
    logging.info('=== Data preprocessing completed! ===')
    logging.info(f'Cache files saved to: {cache_dir}')


if __name__ == '__main__':
    main()
