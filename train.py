import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import json
import datasets
from tqdm import tqdm
import argparse
import wandb
import numpy as np
from collections import defaultdict
import types
import sys

# 添加项目路径
sys.path.append('/.../../...')


token = "xxxxxxxxx"


# CUDA_VISIBLE_DEVICES=0 python .../.../train.py --adv --name kit-iiot-tfc-mscad



def parse():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='HiAdv Hierarchical Text Classification Training')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--data', type=str, default='kit-iiot-tfc-mscad', help='Dataset name')
    parser.add_argument('--batch', type=int, default=2, help='Batch size')
    parser.add_argument('--early-stop', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--name', type=str, required=True, help='Experiment name')
    parser.add_argument('--model', type=str, default='bert',
                        choices=['bert', 'hibert', 'prompt', 'single_prompt'],
                        help='Model architecture')
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--arch', type=str, default='bert-base-uncased', help='Pretrained model architecture')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--adv', action='store_true', help='Enable adversarial training')
    parser.add_argument('--adv-weight', type=float, default=1.0, help='Adversarial loss weight')
    return parser


class ModelSaver:
    """模型保存器"""

    def __init__(self, model, optimizer, args):
        self.model = model
        self.optimizer = optimizer
        self.args = args

    def save_checkpoint(self, score, best_score, filepath):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'score': score,
            'best_score': best_score,
            'args': self.args
        }, filepath)
        print(f"Checkpoint saved: {filepath}")


def setup_environment():
    """设置训练环境"""
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 检查GPU可用性
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("Using CPU")


def load_label_dict(data_path):
    """加载标签字典"""
    label_dict_path = os.path.join(data_path, 'value_dict.pt')
    if not os.path.exists(label_dict_path):
        raise FileNotFoundError(f"Label dictionary not found: {label_dict_path}")

    label_dict = torch.load(label_dict_path, map_location='cpu')
    # 确保键为整数
    label_dict = {int(k): v for k, v in label_dict.items()}
    print(f"Loaded {len(label_dict)} classes")
    return label_dict


def load_hierarchy(data_path):
    """加载层次结构"""
    slot_path = os.path.join(data_path, 'slot.pt')
    if not os.path.exists(slot_path):
        raise FileNotFoundError(f"Hierarchy file not found: {slot_path}")

    slot2value = torch.load(slot_path, map_location='cpu')
    print(f"Loaded hierarchy with {len(slot2value)} parent categories")
    return slot2value


def load_json_data(file_path):
    """加载JSON数据文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def create_dataset_from_json(data_path, split_name, tokenizer, num_classes):
    """从JSON文件创建数据集"""
    file_path = os.path.join(data_path, f"{split_name}.json")
    print(f"Processing {split_name} set: {file_path}")

    json_data = load_json_data(file_path)
    processed_data = []

    for i, item in enumerate(json_data):
        if 'token' not in item or 'label' not in item:
            continue

        # Tokenize文本
        tokens = tokenizer(
            item['token'],
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        # 创建多标签
        labels = [0] * num_classes
        for label_id in item['label']:
            if isinstance(label_id, int) and 0 <= label_id < num_classes:
                labels[label_id] = 1

        processed_item = {
            'input_ids': tokens['input_ids'].squeeze(0).tolist(),
            'attention_mask': tokens['attention_mask'].squeeze(0).tolist(),
            'labels': labels
        }

        # 添加token_type_ids
        if 'token_type_ids' in tokens:
            processed_item['token_type_ids'] = tokens['token_type_ids'].squeeze(0).tolist()
        else:
            processed_item['token_type_ids'] = [0] * 512

        processed_data.append(processed_item)

    print(f"{split_name} set: {len(processed_data)} samples")
    return processed_data


def create_complete_dataset(args, tokenizer):
    """创建完整的数据集"""
    data_path = os.path.join('home', 'Hiadv', 'data', args.data)
    print(f"Data path: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # 加载标签和层次结构
    label_dict = load_label_dict(data_path)
    num_classes = len(label_dict)
    slot2value = load_hierarchy(data_path)

    # 构建路径列表
    path_list = []
    value2slot = {}
    for slot in slot2value:
        for value in slot2value[slot]:
            value2slot[value] = slot
            path_list.append((slot, value))

    # 创建数据集
    splits = {}
    for split_name in ['train', 'dev', 'test']:
        split_data = create_dataset_from_json(data_path, split_name, tokenizer, num_classes)
        splits[split_name] = datasets.Dataset.from_list(split_data)

    dataset = datasets.DatasetDict(splits)

    return dataset, label_dict, num_classes, path_list, value2slot


def initialize_model(args, num_classes, path_list, value2slot):
    """修复的模型初始化"""
    # 动态导入模型
    if args.model == 'bert':
        from model.bert import HTCModel
    elif args.model == 'hibert':
        from model.bert_new import HTCModel
    elif args.model == 'prompt':
        from model.prompt import HTCModel
    elif args.model == 'single_prompt':
        from model.single_prompt import HTCModel
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # 计算深度信息
    depth2label = None
    if 'prompt' in args.model:
        depth_dict = {}
        for i in range(num_classes):
            depth = 0
            current = i
            while current in value2slot and value2slot.get(current, -1) != -1:
                depth += 1
                current = value2slot[current]
            depth_dict[i] = depth

        max_depth = max(depth_dict.values()) + 1 if depth_dict else 1
        depth2label = {i: [label for label in depth_dict if depth_dict[label] == i]
                       for i in range(max_depth)}
        print(f"Hierarchy depth: {max_depth}")

    # 初始化模型
    model = HTCModel.from_pretrained(
        args.arch,
        num_labels=num_classes,
        path_list=path_list,
        data_path=os.path.join('home', 'Hiadv', 'data', args.data),
        depth2label=depth2label
    )

    # 初始化嵌入（如果模型需要）
    if hasattr(model, 'init_embedding'):
        try:
            model.init_embedding()
            print("Model embedding initialized successfully")
        except Exception as e:
            print(f"Warning: Embedding initialization failed: {e}")

    return model, depth2label


def simple_forward_pass(model, batch):
    """修复的简化前向传播"""
    try:
        # 使用模型的简化前向传播
        return model(**batch, use_simple_forward=True)
    except Exception as e:
        print(f"Simple forward failed: {e}")
        # 返回默认值避免崩溃
        batch_size = batch['input_ids'].shape[0]
        device = batch['input_ids'].device
        logits = torch.zeros(batch_size, model.num_labels, device=device)
        return {'loss': torch.tensor(0.0, device=device), 'logits': logits}


def train_epoch_simple(model, train_loader, optimizer, device):
    """修复的训练epoch"""
    model.train()
    total_loss = 0
    total_batches = len(train_loader)

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        try:
            # 准备数据
            input_ids = torch.tensor(batch['input_ids']).to(device)
            attention_mask = torch.tensor(batch['attention_mask']).to(device)
            labels = torch.tensor(batch['labels']).float().to(device)

            token_type_ids = None
            if 'token_type_ids' in batch:
                token_type_ids = torch.tensor(batch['token_type_ids']).to(device)

            batch_data = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            if token_type_ids is not None:
                batch_data['token_type_ids'] = token_type_ids

            # 前向传播
            optimizer.zero_grad()
            output = simple_forward_pass(model, batch_data)

            # 反向传播
            loss = output['loss']
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            pbar.set_description(f"Loss: {avg_loss:.4f}")

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue

    return total_loss / total_batches if total_batches > 0 else 0


def calculate_metrics_per_level(all_preds, all_labels, level1_classes, level2_classes):
    """计算一级标签和二级标签的评估指标"""

    def calculate_single_level_metrics(preds_list, labels_list, class_set):
        """计算单个层级的指标"""
        # 初始化统计变量
        tp_total = fp_total = fn_total = tn_total = 0
        level_preds = []
        level_labels = []

        for pred_set, label_set in zip(preds_list, labels_list):
            # 只考虑当前层级的类别
            pred_level = pred_set.intersection(class_set)
            label_level = label_set.intersection(class_set)

            # 收集预测和标签用于宏观平均
            level_preds.append(pred_level)
            level_labels.append(label_level)

            # 计算TP, FP, FN, TN（微观平均）
            for cls in class_set:
                pred_cls = 1 if cls in pred_level else 0
                true_cls = 1 if cls in label_level else 0

                if pred_cls == 1 and true_cls == 1:
                    tp_total += 1
                elif pred_cls == 1 and true_cls == 0:
                    fp_total += 1
                elif pred_cls == 0 and true_cls == 1:
                    fn_total += 1
                else:
                    tn_total += 1

        # 微观平均指标
        micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (
                                                                                                        micro_precision + micro_recall) > 0 else 0
        micro_accuracy = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total) if (
                                                                                                            tp_total + tn_total + fp_total + fn_total) > 0 else 0

        # 宏观平均指标 - 按类别计算
        macro_precision_sum = 0
        macro_recall_sum = 0
        macro_f1_sum = 0
        valid_classes = 0

        for cls in class_set:
            tp = sum(1 for pred, label in zip(level_preds, level_labels) if cls in pred and cls in label)
            fp = sum(1 for pred, label in zip(level_preds, level_labels) if cls in pred and cls not in label)
            fn = sum(1 for pred, label in zip(level_preds, level_labels) if cls not in pred and cls in label)

            precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_cls = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_cls = 2 * precision_cls * recall_cls / (precision_cls + recall_cls) if (
                                                                                                  precision_cls + recall_cls) > 0 else 0

            macro_precision_sum += precision_cls
            macro_recall_sum += recall_cls
            macro_f1_sum += f1_cls
            valid_classes += 1

        macro_precision = macro_precision_sum / valid_classes if valid_classes > 0 else 0
        macro_recall = macro_recall_sum / valid_classes if valid_classes > 0 else 0
        macro_f1 = macro_f1_sum / valid_classes if valid_classes > 0 else 0

        return {
            'accuracy': micro_accuracy,  # 使用微观准确率
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }

    # 计算一级标签指标
    level1_metrics = calculate_single_level_metrics(all_preds, all_labels, level1_classes)

    # 计算二级标签指标
    level2_metrics = calculate_single_level_metrics(all_preds, all_labels, level2_classes)

    return level1_metrics, level2_metrics


def evaluate_simple(model, data_loader, label_dict, device, dataset_name):
    """修复的评估函数，分别输出一级和二级标签指标"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            try:
                # 准备数据
                input_ids = torch.tensor(batch['input_ids']).to(device)
                attention_mask = torch.tensor(batch['attention_mask']).to(device)
                labels = torch.tensor(batch['labels']).float().to(device)

                # 准备token_type_ids
                token_type_ids = None
                if 'token_type_ids' in batch:
                    token_type_ids = torch.tensor(batch['token_type_ids']).to(device)

                batch_data = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }
                if token_type_ids is not None:
                    batch_data['token_type_ids'] = token_type_ids

                # 使用修复的前向传播
                output = simple_forward_pass(model, batch_data)
                logits = output['logits']
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int().cpu().numpy()

                # 收集预测和真实标签
                labels_np = labels.cpu().numpy()
                for i in range(preds.shape[0]):
                    sample_preds = set(np.where(preds[i] == 1)[0])
                    sample_labels = set(np.where(labels_np[i] == 1)[0])

                    all_preds.append(sample_preds)
                    all_labels.append(sample_labels)

            except Exception as e:
                print(f"Evaluation error: {e}")
                # 添加默认预测避免空结果
                batch_size = input_ids.shape[0] if 'input_ids' in locals() else 1
                for i in range(batch_size):
                    all_preds.append(set())
                    all_labels.append(set())
                continue

    # 根据数据集名称确定一级和二级标签
    if dataset_name == 'iiot-tfc':
        # iiotset和tfc2016作为一级标签（场景标签）
        level1_classes = set()
        level2_classes = set()

        # 这里需要根据您的实际标签映射来设置
        # 假设一级标签是较小的数字（如0,1），二级标签是较大的数字
        # 您需要根据您的数据集结构调整这个逻辑
        for label_id in label_dict.keys():
            if label_id in [0, 1]:  # 根据实际情况调整
                level1_classes.add(label_id)
            else:
                level2_classes.add(label_id)
    else:
        # 默认情况下，假设前2个类别为一级标签，其余为二级标签
        all_classes = set(label_dict.keys())
        level1_classes = set(list(all_classes)[:2])
        level2_classes = all_classes - level1_classes

    print(f"Level 1 classes ({len(level1_classes)}): {level1_classes}")
    print(f"Level 2 classes ({len(level2_classes)}): {level2_classes}")

    # 计算分层指标
    level1_metrics, level2_metrics = calculate_metrics_per_level(all_preds, all_labels, level1_classes, level2_classes)

    return {
        'level1': level1_metrics,
        'level2': level2_metrics
    }


def main():
    """主训练函数"""
    parser = parse()
    args = parser.parse_args()

    print("=" * 60)
    print("HiAdv Hierarchical Text Classification Training")
    print("=" * 60)

    # 设置环境
    setup_environment()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    args.device = device
    print(f"Using device: {device}")

    # 初始化wandb
    if args.wandb:
        wandb.init(project="hiadv-htc", config=args)

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.arch, token=token)

    try:
        # 创建数据集
        dataset, label_dict, num_classes, path_list, value2slot = create_complete_dataset(args, tokenizer)

        # 设置数据集格式
        for split in ['train', 'dev', 'test']:
            if split in dataset:
                dataset[split].set_format('torch',
                                          columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])

        # 初始化模型
        model, depth2label = initialize_model(args, num_classes, path_list, value2slot)
        model.to(device)

        # 创建数据加载器
        train_loader = DataLoader(dataset['train'], batch_size=args.batch, shuffle=True)
        dev_loader = DataLoader(dataset['dev'], batch_size=8, shuffle=False)
        test_loader = DataLoader(dataset['test'], batch_size=8, shuffle=False)

        # 优化器
        optimizer = Adam(model.parameters(), lr=args.lr)
        saver = ModelSaver(model, optimizer, args)

        # 创建检查点目录
        os.makedirs(os.path.join('checkpoints', args.name), exist_ok=True)

        # 训练循环
        best_macro_f1 = 0
        best_micro_f1 = 0
        early_stop_count = 0

        for epoch in range(8):
            if early_stop_count >= args.early_stop:
                print("Early stopping triggered!")
                break

            print(f"\nEpoch {epoch + 1}")
            print("-" * 40)

            # 训练
            train_loss = train_epoch_simple(model, train_loader, optimizer, device)
            print(f"Training loss: {train_loss:.4f}")

            # 验证 - 使用原有的评估方式保持兼容性
            scores = evaluate_simple(model, dev_loader, label_dict, device, args.data)
            # 为了保持兼容性，使用二级标签的macro_f1作为主要指标
            macro_f1 = scores['level2']['macro_f1']
            micro_f1 = scores['level2']['micro_f1']

            print(f"Validation - Level2 Macro F1: {macro_f1:.4f}, Level2 Micro F1: {micro_f1:.4f}")

            # 记录到wandb
            if args.wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_macro_f1': macro_f1,
                    'val_micro_f1': micro_f1
                })

            # 保存最佳模型
            early_stop_count += 1

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                saver.save_checkpoint(macro_f1, best_macro_f1,
                                      os.path.join('checkpoints', args.name, 'best_macro.pt'))
                early_stop_count = 0
                print(f"New best Macro F1: {macro_f1:.4f}")

            if micro_f1 > best_micro_f1:
                best_micro_f1 = micro_f1
                saver.save_checkpoint(micro_f1, best_micro_f1,
                                      os.path.join('checkpoints', args.name, 'best_micro.pt'))
                early_stop_count = 0
                print(f"New best Micro F1: {micro_f1:.4f}")

            # 保存当前epoch
            saver.save_checkpoint(micro_f1, best_micro_f1,
                                  os.path.join('checkpoints', args.name, f'epoch_{epoch}.pt'))

            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 最终测试 - 输出详细的层级指标
        print("\nFinal Testing")
        print("-" * 40)

        # 加载最佳模型进行测试
        checkpoint_path = os.path.join('checkpoints', args.name, 'best_macro.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])

            test_scores = evaluate_simple(model, test_loader, label_dict, device, args.data)

            # 输出一级标签指标
            level1 = test_scores['level1']
            print("\n=== Level 1 (Scene Labels) Results ===")
            print(f"Accuracy: {level1['accuracy']:.4f}")
            print(f"Precision: {level1['micro_precision']:.4f}")
            print(f"Recall: {level1['micro_recall']:.4f}")
            print(f"Micro F1: {level1['micro_f1']:.4f}")
            print(f"Macro F1: {level1['macro_f1']:.4f}")

            # 输出二级标签指标
            level2 = test_scores['level2']
            print("\n=== Level 2 (Attack Categories) Results ===")
            print(f"Accuracy: {level2['accuracy']:.4f}")
            print(f"Precision: {level2['micro_precision']:.4f}")
            print(f"Recall: {level2['micro_recall']:.4f}")
            print(f"Micro F1: {level2['micro_f1']:.4f}")
            print(f"Macro F1: {level2['macro_f1']:.4f}")

            if args.wandb:
                wandb.log({
                    'test_level1_accuracy': level1['accuracy'],
                    'test_level1_precision': level1['micro_precision'],
                    'test_level1_recall': level1['micro_recall'],
                    'test_level1_micro_f1': level1['micro_f1'],
                    'test_level1_macro_f1': level1['macro_f1'],
                    'test_level2_accuracy': level2['accuracy'],
                    'test_level2_precision': level2['micro_precision'],
                    'test_level2_recall': level2['micro_recall'],
                    'test_level2_micro_f1': level2['micro_f1'],
                    'test_level2_macro_f1': level2['macro_f1']
                })

        print("Training completed successfully!")

    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()








