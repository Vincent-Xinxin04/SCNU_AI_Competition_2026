import os
import csv
import json
import time
import openai
import argparse
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

openai.api_key = os.environ.get("OPENAI_API_KEY", "")


def load_labels(train_dir):
    """从训练集目录加载所有标签（关系类型）"""
    labels = []
    for filename in os.listdir(train_dir):
        if filename.endswith('.csv'):
            label_name = filename[:-4]
            labels.append(label_name)
    return labels


def build_relation_prompt(subject, object_, predicted_relation, all_relations, max_relations=50):
    """构建GPT-4o关系验证提示"""
    relations_list = ", ".join(all_relations[:max_relations])
    if len(all_relations) > max_relations:
        relations_list += f", ... and {len(all_relations) - max_relations} more"

    prompt = f"""You are an expert at understanding relationships between entities.

Given a Subject, Object, and a Predicted Relation, your task is to:
1. First, evaluate if the Predicted Relation is CORRECT for this Subject-Object pair
2. If INCORRECT, provide the CORRECT relation from the list below
3. If CORRECT, respond with "CORRECT"

Available relations: {relations_list}

Subject: {subject}
Object: {object_}
Predicted Relation: {predicted_relation}

Respond in exactly this JSON format:
{{"status": "CORRECT" or "INCORRECT", "corrected_relation": "the correct relation if status is INCORRECT, otherwise null"}}

Important:
- Only respond with valid JSON, no other text
- If the relation is incorrect, choose ONE relation from the available list that best describes the relationship
- Be precise and consider the semantic meaning of the Subject-Object relationship"""

    return prompt


def verify_with_gpt4o(subject, object_, predicted_relation, all_relations, model="gpt-4o", max_retries=3):
    """使用GPT-4o验证并纠正关系预测"""
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    prompt = build_relation_prompt(subject, object_, predicted_relation, all_relations)

    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise relation classification assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100,
                timeout=30
            )

            content = response.choices[0].message.content.strip()

            result = json.loads(content)

            return {
                "original": predicted_relation,
                "status": result.get("status", "CORRECT"),
                "corrected": result.get("corrected_relation", None),
                "final": predicted_relation if result.get("status") == "CORRECT" else result.get("corrected_relation", predicted_relation)
            }

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            return {
                "original": predicted_relation,
                "status": "ERROR",
                "corrected": None,
                "final": predicted_relation
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return {
                "original": predicted_relation,
                "status": "ERROR",
                "corrected": None,
                "final": predicted_relation
            }

    return {
        "original": predicted_relation,
        "status": "ERROR",
        "corrected": None,
        "final": predicted_relation
    }


def verify_single_item(args):
    """单个项目的验证（用于并行处理）"""
    idx, subject, object_, label, all_relations, model = args
    result = verify_with_gpt4o(subject, object_, label, all_relations, model)
    result["index"] = idx
    result["subject"] = subject
    result["object"] = object_
    return result


def correct_submission(input_path, output_path, train_dir, model="gpt-4o", max_workers=10, delay=0.5):
    """读取submission文件并用GPT-4o验证纠正"""
    print(f"Loading labels from {train_dir}...")
    all_relations = load_labels(train_dir)
    print(f"Found {len(all_relations)} relation types")

    print(f"Loading submission from {input_path}...")
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    df.columns = [str(col).strip() for col in df.columns]

    subject_col = None
    object_col = None
    label_col = None
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'subject':
            subject_col = col
        elif col_lower == 'object':
            object_col = col
        elif col_lower == 'label':
            label_col = col

    if subject_col is None or object_col is None:
        raise ValueError("CSV must contain 'Subject' and 'Object' columns")

    if label_col is None:
        label_col = 'Label'
        df['Label'] = 'unknown'

    valid_mask = df[subject_col].notna() & df[object_col].notna()
    valid_indices = df[valid_mask].index.tolist()

    print(f"Total rows: {len(df)}, Valid rows for correction: {len(valid_indices)}")

    corrected_count = 0
    results = {}

    tasks = [
        (idx, str(df.loc[idx, subject_col]), str(df.loc[idx, object_col]),
         str(df.loc[idx, label_col]), all_relations, model)
        for idx in valid_indices
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(verify_single_item, task): task for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="GPT-4o Verification"):
            result = future.result()
            idx = result["index"]
            results[idx] = result

            if result["status"] == "INCORRECT":
                corrected_count += 1
                df.loc[idx, label_col] = result["final"]
            elif result["status"] == "CORRECT":
                pass

            time.sleep(delay)

    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"\nCorrection Summary:")
    print(f"  Total samples: {len(valid_indices)}")
    print(f"  Correct (kept): {len(valid_indices) - corrected_count}")
    print(f"  Incorrect (corrected): {corrected_count}")
    print(f"  Correction rate: {corrected_count / len(valid_indices) * 100:.2f}%")
    print(f"\nResults saved to: {output_path}")

    correction_log = output_path.replace('.csv', '_correction_log.json')
    with open(correction_log, 'w', encoding='utf-8') as f:
        log_data = {
            "total": len(valid_indices),
            "corrected": corrected_count,
            "correction_rate": corrected_count / len(valid_indices) * 100,
            "details": [
                {
                    "index": int(r["index"]),
                    "subject": r["subject"],
                    "object": r["object"],
                    "original": r["original"],
                    "status": r["status"],
                    "corrected": r["corrected"]
                }
                for r in results.values() if r["status"] == "INCORRECT"
            ]
        }
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"Correction log saved to: {correction_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-4o Relation Correction")
    parser.add_argument("--input", type=str, default="./result/submission-0.717.csv",
                        help="Input submission CSV file")
    parser.add_argument("--output", type=str, default="./result/submission-gpt4o-corrected.csv",
                        help="Output corrected CSV file")
    parser.add_argument("--train_dir", type=str, default="./dataset/Train_Set",
                        help="Training data directory (for loading relation types)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="GPT model to use (gpt-4o, gpt-4-turbo, etc.)")
    parser.add_argument("--max_workers", type=int, default=10,
                        help="Number of parallel API calls")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API calls in seconds")

    args = parser.parse_args()
    correct_submission(args.input, args.output, args.train_dir, args.model, args.max_workers, args.delay)
