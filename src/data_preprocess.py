import argparse
import os
import logging
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[logging.StreamHandler()]
    )


def load_data_from_directory(dir_path):
    """加载目录下所有CSV文件"""
    all_data = []
    if not os.path.exists(dir_path):
        raise ValueError(f"Directory not found: {dir_path}")

    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    logging.info(f"Loading data from {dir_path}...")

    for filename in tqdm(csv_files, desc=f"Loading {os.path.basename(dir_path)}"):
        file_path = os.path.join(dir_path, filename)
        label_name = filename[:-4]
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding='utf-8-sig')
            if df.empty:
                continue
            df.columns = [str(col).strip() for col in df.columns]
            if 'Subject' in df.columns and 'Object' in df.columns:
                df = df[['Subject', 'Object']].dropna()
                df['label'] = label_name
                all_data.append(df)
        except Exception as e:
            logging.warning(f"{filename} load error: {e}")

    if not all_data:
        raise ValueError(f"No valid data found in {dir_path}")

    full_df = pd.concat(all_data, ignore_index=True)
    full_df['Subject'] = full_df['Subject'].astype(str)
    full_df['Object'] = full_df['Object'].astype(str)
    return full_df


def detect_poisoned_data(df, threshold_ratio=0.95, min_samples=5):
    """检测并移除中毒数据"""
    label_counts = df['label'].value_counts()
    rows_to_remove = []

    for label in tqdm(label_counts.index, desc="Detecting poisoned data"):
        label_df = df[df['label'] == label]
        n_samples = len(label_df)
        
        if n_samples < min_samples:
            continue
        
        # 检测Subject重复率
        subject_counts = label_df['Subject'].value_counts()
        if len(subject_counts) > 0:
            max_subject_ratio = subject_counts.iloc[0] / n_samples
            if max_subject_ratio > threshold_ratio:
                logging.warning(f"Removing poisoned label '{label}' (Subject repetition: {max_subject_ratio:.4f})")
                rows_to_remove.extend(label_df.index.tolist())
                continue
        
        # 检测Object重复率
        object_counts = label_df['Object'].value_counts()
        if len(object_counts) > 0:
            max_object_ratio = object_counts.iloc[0] / n_samples
            if max_object_ratio > threshold_ratio:
                logging.warning(f"Removing poisoned label '{label}' (Object repetition: {max_object_ratio:.4f})")
                rows_to_remove.extend(label_df.index.tolist())
                continue
        
        # 检测(S,O)对重复率
        pair_counts = label_df.groupby(['Subject', 'Object']).size().sort_values(ascending=False)
        if len(pair_counts) > 0:
            max_pair_ratio = pair_counts.iloc[0] / n_samples
            if max_pair_ratio > threshold_ratio:
                logging.warning(f"Removing poisoned label '{label}' (Pair repetition: {max_pair_ratio:.4f})")
                rows_to_remove.extend(label_df.index.tolist())
                continue

    cleaned_df = df.drop(rows_to_remove).reset_index(drop=True)
    removed_count = len(rows_to_remove)
    logging.info(f"Removed {removed_count} poisoned samples")
    return cleaned_df


def detect_outlier_samples(df, z_score_threshold=3):
    """检测异常样本（基于文本长度）"""
    df['subject_len'] = df['Subject'].apply(len)
    df['object_len'] = df['Object'].apply(len)
    df['total_len'] = df['subject_len'] + df['object_len']
    
    mean_len = df['total_len'].mean()
    std_len = df['total_len'].std()
    
    if std_len == 0:
        return df.drop(['subject_len', 'object_len', 'total_len'], axis=1), 0
    
    df['z_score'] = (df['total_len'] - mean_len) / std_len
    outliers = (df['z_score'].abs() > z_score_threshold)
    outlier_count = outliers.sum()
    
    if outlier_count > 0:
        logging.warning(f"Found {outlier_count} outlier samples")
    
    cleaned_df = df[~outliers].drop(['subject_len', 'object_len', 'total_len', 'z_score'], axis=1).reset_index(drop=True)
    return cleaned_df, outlier_count


def analyze_data_distribution(df):
    """分析数据分布"""
    label_counts = df['label'].value_counts()
    
    logging.info(f"\n=== Data Distribution Analysis ===")
    logging.info(f"Total samples: {len(df)}")
    logging.info(f"Total labels: {len(label_counts)}")
    logging.info(f"Min samples per label: {label_counts.min()}")
    logging.info(f"Max samples per label: {label_counts.max()}")
    logging.info(f"Mean samples per label: {label_counts.mean():.2f}")
    logging.info(f"Std samples per label: {label_counts.std():.2f}")
    
    few_shot_labels = label_counts[label_counts < 5].index
    logging.info(f"Few-shot labels (<5 samples): {len(few_shot_labels)}")
    
    frequent_labels = label_counts[label_counts > 100].index
    logging.info(f"Frequent labels (>100 samples): {len(frequent_labels)}")
    
    return label_counts


def calculate_label_weights(label_counts):
    """根据赛题公式计算少样本重要性权重"""
    counts_max = label_counts.max()
    counts_min = label_counts.min()
    
    weights = {}
    for label, count in label_counts.items():
        numerator = counts_max - count + counts_min * 0.1
        denominator = counts_max + counts_min * 0.1
        weights[label] = numerator / denominator
    
    logging.info(f"\n=== Weight Calculation ===")
    logging.info(f"counts_max: {counts_max}")
    logging.info(f"counts_min: {counts_min}")
    logging.info(f"Min weight: {min(weights.values()):.4f}")
    logging.info(f"Max weight: {max(weights.values()):.4f}")
    
    return weights


def save_cleaned_data(df, output_dir, filename='cleaned_train_data.csv'):
    """保存清理后的数据"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logging.info(f"Cleaned data saved to: {output_path}")


def main(args):
    setup_logging()
    
    # 1. 加载数据
    logging.info("Step 1: Loading raw data...")
    raw_df = load_data_from_directory(args.train_dir)
    
    # 2. 分析原始数据分布
    logging.info("Step 2: Analyzing raw data distribution...")
    analyze_data_distribution(raw_df)
    
    # 3. 检测并移除中毒数据
    logging.info("\nStep 3: Detecting poisoned data...")
    cleaned_df = detect_poisoned_data(
        raw_df, 
        threshold_ratio=args.poison_threshold,
        min_samples=args.min_samples
    )
    
    # 4. 检测并移除异常样本
    logging.info("\nStep 4: Detecting outlier samples...")
    cleaned_df, outlier_count = detect_outlier_samples(cleaned_df, z_score_threshold=args.z_threshold)
    
    # 5. 分析清理后的数据分布
    logging.info("\nStep 5: Analyzing cleaned data distribution...")
    cleaned_counts = analyze_data_distribution(cleaned_df)
    
    # 6. 计算权重
    logging.info("\nStep 6: Calculating weights...")
    weights = calculate_label_weights(cleaned_counts)
    
    # 7. 保存清理后的数据
    logging.info("\nStep 7: Saving cleaned data...")
    save_cleaned_data(cleaned_df, args.output_dir)
    
    # 8. 保存权重文件
    weights_df = pd.DataFrame(list(weights.items()), columns=['label', 'weight'])
    weights_df.to_csv(os.path.join(args.output_dir, 'label_weights.csv'), index=False, encoding='utf-8-sig')
    logging.info("Label weights saved to: label_weights.csv")
    
    logging.info("\n=== Preprocessing Complete ===")
    logging.info(f"Original samples: {len(raw_df)}")
    logging.info(f"Cleaned samples: {len(cleaned_df)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Preprocessing for CPA Task')
    parser.add_argument('--train_dir', type=str, default="./dataset/Train_Set")
    parser.add_argument('--output_dir', type=str, default="./dataset")
    parser.add_argument('--poison_threshold', type=float, default=0.95)
    parser.add_argument('--min_samples', type=int, default=5)
    parser.add_argument('--z_threshold', type=float, default=3.0)
    
    args = parser.parse_args()
    main(args)