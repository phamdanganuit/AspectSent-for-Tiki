import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.constants import GOLD_DIR, LOGS_DIR

# Tạo thư mục thời gian cho lần chạy hiện tại
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_LOG_DIR = os.path.join(LOGS_DIR, f"finetune_{current_time}")
os.makedirs(RUN_LOG_DIR, exist_ok=True)

# Thư mục mô hình
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# File dữ liệu đầu vào từ bước embedding
INPUT_IDS_FILE = os.path.join(GOLD_DIR, "input_ids.pt")
ATTENTION_MASKS_FILE = os.path.join(GOLD_DIR, "attention_masks.pt")
SENTIMENT_LABELS_FILE = os.path.join(GOLD_DIR, "sentiment_labels.pt")
ASPECT_LABELS_FILE = os.path.join(GOLD_DIR, "aspect_labels.pt")
FINETUNING_METADATA_FILE = os.path.join(GOLD_DIR, "finetuning_metadata.csv")

# Tên file đầu ra
FINETUNED_MODEL_DIR = os.path.join(MODELS_DIR, "phobert_finetuned")
os.makedirs(FINETUNED_MODEL_DIR, exist_ok=True)

# Thông số huấn luyện
BATCH_SIZE = 8
MAX_EPOCHS = 10  # Số epoch tối đa
LEARNING_RATE = 2e-5
NUM_SENTIMENT_CLASSES = 5  # 0-4 (5 classes)
NUM_ASPECT_CLASSES = 5  # Dựa vào ASPECT_MAPPING trong phobert_embedding.py
SEED = 42
MAX_LENGTH = 256
PATIENCE = 3  # Số epoch kiên nhẫn cho early stopping
ASPECT_MAPPING = {0: 'other', 1: 'cskh', 2: 'quality', 3: 'price', 4: 'ship'}
SENTIMENT_MAPPING = {0: 'rất tiêu cực', 1: 'tiêu cực', 2: 'trung lập', 3: 'tích cực', 4: 'rất tích cực'}

# Thiết lập logging
def setup_logger():
    import logging
    log_file = os.path.join(RUN_LOG_DIR, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# Thiết lập seed cho khả năng tái tạo kết quả
def set_seed(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Đã thiết lập seed: {seed_value}")

# Dataset cho việc fine-tuning đa nhiệm vụ
class MultiTaskDataset(Dataset):
    def __init__(self, input_ids, attention_masks, sentiment_labels, aspect_labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.sentiment_labels = sentiment_labels
        self.aspect_labels = aspect_labels
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'sentiment_labels': self.sentiment_labels[idx],
            'aspect_labels': self.aspect_labels[idx]
        }

# Khối CNN 1D
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 4, 5]):
        super(CNNBlock, self).__init__()
        
        # Các lớp CNN với các kích thước kernel khác nhau
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels, 
                     out_channels=out_channels, 
                     kernel_size=k) 
            for k in kernel_sizes
        ])
        
    def forward(self, x):
        # x: batch_size x sequence_length x embedding_dim
        
        # Chuyển đổi sang định dạng cho CNN: batch_size x embedding_dim x sequence_length
        x = x.permute(0, 2, 1)
        
        # Áp dụng các lớp CNN và max-over-time pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x)
            conv_out = torch.relu(conv_out)
            # Max pooling
            pool_out = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pool_out)
        
        # Ghép các đầu ra từ các kích thước kernel khác nhau
        combined = torch.cat(conv_outputs, dim=1)
        return combined

# Mô hình đa nhiệm vụ dựa trên PhoBERT kết hợp CNN
class MultiTaskPhoBERTCNN(nn.Module):
    def __init__(self, num_sentiment_classes, num_aspect_classes):
        super(MultiTaskPhoBERTCNN, self).__init__()
        
        # Tải pretrained PhoBERT
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        
        # Lớp dropout
        self.dropout = nn.Dropout(0.1)
        
        # Kích thước ẩn của PhoBERT
        hidden_size = self.phobert.config.hidden_size
        
        # Mạng CNN cho trích xuất đặc trưng từ chuỗi đầu ra PhoBERT
        self.cnn_block = CNNBlock(in_channels=hidden_size, out_channels=128)
        
        # Kích thước đầu ra của CNN (128 cho mỗi kích thước kernel, 3 kích thước kernel)
        cnn_output_size = 128 * 3
        
        # Các lớp phân loại
        # Nhiệm vụ 1: Phân loại sentiment (multi-class)
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size + cnn_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_sentiment_classes)
        )
        
        # Nhiệm vụ 2: Phân loại aspect (multi-label)
        self.aspect_classifier = nn.Sequential(
            nn.Linear(hidden_size + cnn_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_aspect_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # Lấy đầu ra từ PhoBERT
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Lấy embedding của token [CLS] (đầu tiên)
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Lấy đầu ra chuỗi từ PhoBERT để đưa vào CNN
        sequence_output = outputs.last_hidden_state
        
        # Trích xuất đặc trưng bằng CNN
        cnn_features = self.cnn_block(sequence_output)
        
        # Kết hợp đặc trưng từ [CLS] và CNN
        combined_features = torch.cat([cls_output, cnn_features], dim=1)
        combined_features = self.dropout(combined_features)
        
        # Dự đoán cho từng nhiệm vụ
        sentiment_logits = self.sentiment_classifier(combined_features)
        aspect_logits = self.aspect_classifier(combined_features)
        
        return sentiment_logits, aspect_logits

# Hàm đánh giá cho nhiệm vụ phân loại sentiment
def evaluate_sentiment(predictions, true_labels):
    preds_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = true_labels.flatten()
    
    return {
        'accuracy': accuracy_score(labels_flat, preds_flat),
        'f1': f1_score(labels_flat, preds_flat, average='weighted'),
        'precision': precision_score(labels_flat, preds_flat, average='weighted'),
        'recall': recall_score(labels_flat, preds_flat, average='weighted')
    }

# Hàm đánh giá cho nhiệm vụ phân loại aspect
def evaluate_aspect(predictions, true_labels, threshold=0.5):
    # Áp dụng sigmoid và ngưỡng để chuyển đổi logits thành dự đoán nhị phân
    preds = torch.sigmoid(torch.Tensor(predictions)).numpy()
    preds_binary = (preds >= threshold).astype(int)
    
    return {
        'f1_micro': f1_score(true_labels, preds_binary, average='micro'),
        'f1_macro': f1_score(true_labels, preds_binary, average='macro'),
        'precision': precision_score(true_labels, preds_binary, average='micro', zero_division=0),
        'recall': recall_score(true_labels, preds_binary, average='micro', zero_division=0)
    }

# Vẽ biểu đồ quá trình huấn luyện
def plot_training_metrics(metrics, title, filename):
    plt.figure(figsize=(12, 8))
    
    # Số epoch
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    # Vẽ loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Vẽ sentiment metrics
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics['val_sentiment_accuracy'], label='Accuracy')
    plt.plot(epochs, metrics['val_sentiment_f1'], label='F1')
    plt.plot(epochs, metrics['val_sentiment_precision'], label='Precision')
    plt.plot(epochs, metrics['val_sentiment_recall'], label='Recall')
    plt.title('Sentiment Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    # Vẽ aspect metrics
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics['val_aspect_f1_micro'], label='F1 Micro')
    plt.plot(epochs, metrics['val_aspect_f1_macro'], label='F1 Macro')
    plt.plot(epochs, metrics['val_aspect_precision'], label='Precision')
    plt.plot(epochs, metrics['val_aspect_recall'], label='Recall')
    plt.title('Aspect Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    # Tối ưu layout
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    # Lưu biểu đồ
    plt.savefig(filename)
    logger.info(f"Đã lưu biểu đồ tại: {filename}")
    plt.close()

# Vẽ biểu đồ confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Sentiment Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Đã lưu confusion matrix tại: {filename}")
    plt.close()

# Hàm huấn luyện mô hình với early stopping
def train_model(model, train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler, device, max_epochs=10, patience=3):
    # Các hàm mất mát
    sentiment_criterion = nn.CrossEntropyLoss()
    aspect_criterion = nn.BCEWithLogitsLoss()
    
    # Lưu các metric tốt nhất
    best_val_f1 = 0
    best_epoch = 0
    no_improve_count = 0
    
    # Theo dõi quá trình huấn luyện
    train_stats = {
        'train_loss': [],
        'val_sentiment_accuracy': [],
        'val_sentiment_f1': [],
        'val_sentiment_precision': [],
        'val_sentiment_recall': [],
        'val_aspect_f1_micro': [],
        'val_aspect_f1_macro': [],
        'val_aspect_precision': [],
        'val_aspect_recall': []
    }
    
    # Training loop
    for epoch in range(max_epochs):
        logger.info(f"\nEpoch {epoch+1}/{max_epochs}")
        
        # Training
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            # Đưa dữ liệu lên GPU/CPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)
            aspect_labels = batch['aspect_labels'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            sentiment_logits, aspect_logits = model(input_ids, attention_mask)
            
            # Tính loss
            sentiment_loss = sentiment_criterion(sentiment_logits, sentiment_labels)
            aspect_loss = aspect_criterion(aspect_logits, aspect_labels)
            
            # Tổng hợp loss (có thể điều chỉnh trọng số nếu cần)
            loss = sentiment_loss + aspect_loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradient norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_dataloader)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        train_stats['train_loss'].append(avg_train_loss)
        
        # Đánh giá trên tập validation
        model.eval()
        sentiment_preds, aspect_preds = [], []
        sentiment_labels_list, aspect_labels_list = [], []
        
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)
            aspect_labels = batch['aspect_labels'].to(device)
            
            with torch.no_grad():
                sentiment_logits, aspect_logits = model(input_ids, attention_mask)
            
            sentiment_preds.extend(sentiment_logits.detach().cpu().numpy())
            aspect_preds.extend(aspect_logits.detach().cpu().numpy())
            sentiment_labels_list.extend(sentiment_labels.detach().cpu().numpy())
            aspect_labels_list.extend(aspect_labels.detach().cpu().numpy())
        
        # Đánh giá các dự đoán
        sentiment_metrics = evaluate_sentiment(
            np.array(sentiment_preds), 
            np.array(sentiment_labels_list)
        )
        
        aspect_metrics = evaluate_aspect(
            np.array(aspect_preds), 
            np.array(aspect_labels_list)
        )
        
        # Log metrics
        logger.info(f"Sentiment - Accuracy: {sentiment_metrics['accuracy']:.4f}, F1: {sentiment_metrics['f1']:.4f}")
        logger.info(f"Sentiment - Precision: {sentiment_metrics['precision']:.4f}, Recall: {sentiment_metrics['recall']:.4f}")
        logger.info(f"Aspect - F1 Micro: {aspect_metrics['f1_micro']:.4f}, F1 Macro: {aspect_metrics['f1_macro']:.4f}")
        logger.info(f"Aspect - Precision: {aspect_metrics['precision']:.4f}, Recall: {aspect_metrics['recall']:.4f}")
        
        # Lưu metric
        train_stats['val_sentiment_accuracy'].append(sentiment_metrics['accuracy'])
        train_stats['val_sentiment_f1'].append(sentiment_metrics['f1'])
        train_stats['val_sentiment_precision'].append(sentiment_metrics['precision'])
        train_stats['val_sentiment_recall'].append(sentiment_metrics['recall'])
        train_stats['val_aspect_f1_micro'].append(aspect_metrics['f1_micro'])
        train_stats['val_aspect_f1_macro'].append(aspect_metrics['f1_macro'])
        train_stats['val_aspect_precision'].append(aspect_metrics['precision'])
        train_stats['val_aspect_recall'].append(aspect_metrics['recall'])
        
        # Lưu mô hình tốt nhất dựa trên F1 trung bình
        current_val_f1 = (sentiment_metrics['f1'] + aspect_metrics['f1_micro']) / 2
        
        if current_val_f1 > best_val_f1:
            logger.info(f"Cải thiện F1 trung bình từ {best_val_f1:.4f} lên {current_val_f1:.4f}")
            best_val_f1 = current_val_f1
            best_epoch = epoch
            no_improve_count = 0
            
            # Lưu mô hình tốt nhất
            torch.save(model.state_dict(), os.path.join(FINETUNED_MODEL_DIR, "best_model.pt"))
            logger.info(f"Đã lưu mô hình tốt nhất tại epoch {epoch+1}")
            
            # Lưu tokenizer để sử dụng sau này
            tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            tokenizer.save_pretrained(FINETUNED_MODEL_DIR)
        else:
            no_improve_count += 1
            logger.info(f"Không có cải thiện. Đếm: {no_improve_count}/{patience}")
            
            if no_improve_count >= patience:
                logger.info(f"Early stopping tại epoch {epoch+1}")
                break
    
    # Lưu kết quả đánh giá
    pd.DataFrame(train_stats).to_csv(os.path.join(RUN_LOG_DIR, "training_metrics.csv"), index=False, encoding="utf-8")
    
    # Vẽ biểu đồ huấn luyện
    plot_training_metrics(
        train_stats, 
        f'Training Metrics (Best Epoch: {best_epoch+1})',
        os.path.join(RUN_LOG_DIR, "training_metrics.png")
    )
    
    # Kiểm tra trên tập test
    logger.info("\nĐánh giá trên tập test...")
    
    # Tải mô hình tốt nhất
    model.load_state_dict(torch.load(os.path.join(FINETUNED_MODEL_DIR, "best_model.pt")))
    model.eval()
    
    sentiment_preds, aspect_preds = [], []
    sentiment_labels_list, aspect_labels_list = [], []
    
    for batch in tqdm(test_dataloader, desc="Testing"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiment_labels = batch['sentiment_labels'].to(device)
        aspect_labels = batch['aspect_labels'].to(device)
        
        with torch.no_grad():
            sentiment_logits, aspect_logits = model(input_ids, attention_mask)
        
        sentiment_preds.extend(sentiment_logits.detach().cpu().numpy())
        aspect_preds.extend(aspect_logits.detach().cpu().numpy())
        sentiment_labels_list.extend(sentiment_labels.detach().cpu().numpy())
        aspect_labels_list.extend(aspect_labels.detach().cpu().numpy())
    
    # Chuyển đổi dự đoán sentiment thành nhãn
    sentiment_pred_labels = np.argmax(np.array(sentiment_preds), axis=1)
    sentiment_true_labels = np.array(sentiment_labels_list).flatten()
    
    # Tạo classification report cho sentiment
    sentiment_report = classification_report(
        sentiment_true_labels, 
        sentiment_pred_labels,
        target_names=[SENTIMENT_MAPPING[i] for i in range(NUM_SENTIMENT_CLASSES)],
        output_dict=True
    )
    
    # Chuyển đổi dự đoán aspect thành nhãn
    aspect_pred_probs = torch.sigmoid(torch.Tensor(aspect_preds)).numpy()
    aspect_pred_labels = (aspect_pred_probs >= 0.5).astype(int)
    aspect_true_labels = np.array(aspect_labels_list)
    
    # Tạo aspect report cho từng nhãn
    aspect_report = {}
    for i in range(NUM_ASPECT_CLASSES):
        aspect_name = ASPECT_MAPPING[i]
        aspect_report[aspect_name] = {
            'precision': precision_score(aspect_true_labels[:, i], aspect_pred_labels[:, i], zero_division=0),
            'recall': recall_score(aspect_true_labels[:, i], aspect_pred_labels[:, i], zero_division=0),
            'f1-score': f1_score(aspect_true_labels[:, i], aspect_pred_labels[:, i], zero_division=0),
            'support': np.sum(aspect_true_labels[:, i])
        }
    
    # Lưu báo cáo đánh giá
    with open(os.path.join(RUN_LOG_DIR, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write("=== BÁO CÁO ĐÁNH GIÁ TẬP TEST ===\n\n")
        
        f.write("--- Kết quả phân loại SENTIMENT ---\n")
        f.write(classification_report(
            sentiment_true_labels, 
            sentiment_pred_labels,
            target_names=[SENTIMENT_MAPPING[i] for i in range(NUM_SENTIMENT_CLASSES)]
        ))
        
        f.write("\n--- Kết quả phân loại ASPECT ---\n")
        f.write(f"F1 Score (micro): {f1_score(aspect_true_labels, aspect_pred_labels, average='micro'):.4f}\n")
        f.write(f"F1 Score (macro): {f1_score(aspect_true_labels, aspect_pred_labels, average='macro'):.4f}\n\n")
        
        f.write("Chi tiết theo từng aspect:\n")
        for i in range(NUM_ASPECT_CLASSES):
            aspect_name = ASPECT_MAPPING[i]
            precision = precision_score(aspect_true_labels[:, i], aspect_pred_labels[:, i], zero_division=0)
            recall = recall_score(aspect_true_labels[:, i], aspect_pred_labels[:, i], zero_division=0)
            f1 = f1_score(aspect_true_labels[:, i], aspect_pred_labels[:, i], zero_division=0)
            support = np.sum(aspect_true_labels[:, i])
            
            f.write(f"{aspect_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Support: {support}\n")
    
    # Vẽ confusion matrix cho sentiment
    plot_confusion_matrix(
        sentiment_true_labels,
        sentiment_pred_labels,
        classes=[SENTIMENT_MAPPING[i] for i in range(NUM_SENTIMENT_CLASSES)],
        filename=os.path.join(RUN_LOG_DIR, "confusion_matrix.png")
    )
    
    # Lưu metrics dưới dạng JSON
    test_metrics = {
        'sentiment': {
            'accuracy': accuracy_score(sentiment_true_labels, sentiment_pred_labels),
            'f1_weighted': f1_score(sentiment_true_labels, sentiment_pred_labels, average='weighted'),
            'precision_weighted': precision_score(sentiment_true_labels, sentiment_pred_labels, average='weighted'),
            'recall_weighted': recall_score(sentiment_true_labels, sentiment_pred_labels, average='weighted'),
            'report': sentiment_report
        },
        'aspect': {
            'f1_micro': f1_score(aspect_true_labels, aspect_pred_labels, average='micro'),
            'f1_macro': f1_score(aspect_true_labels, aspect_pred_labels, average='macro'),
            'precision_micro': precision_score(aspect_true_labels, aspect_pred_labels, average='micro', zero_division=0),
            'recall_micro': recall_score(aspect_true_labels, aspect_pred_labels, average='micro', zero_division=0),
            'report': aspect_report
        }
    }
    
    # Lưu test metrics
    pd.DataFrame({
        'metric': list(test_metrics['sentiment'].keys())[:-1] + list(test_metrics['aspect'].keys())[:-1],
        'value': [test_metrics['sentiment'][k] for k in list(test_metrics['sentiment'].keys())[:-1]] + 
                [test_metrics['aspect'][k] for k in list(test_metrics['aspect'].keys())[:-1]]
    }).to_csv(os.path.join(RUN_LOG_DIR, "test_metrics.csv"), index=False, encoding="utf-8")
    
    logger.info(f"Đã lưu báo cáo đánh giá chi tiết vào {os.path.join(RUN_LOG_DIR, 'test_report.txt')}")
    logger.info(f"Đánh giá mô hình hoàn tất!")
    
    return model, train_stats, test_metrics

def load_and_prepare_data():
    """
    Tải dữ liệu cho fine-tuning từ các file đã chuẩn bị và chia thành train, val, test
    """
    logger.info("Đang tải dữ liệu cho fine-tuning...")
    
    try:
        input_ids = torch.load(INPUT_IDS_FILE)
        attention_masks = torch.load(ATTENTION_MASKS_FILE)
        sentiment_labels = torch.load(SENTIMENT_LABELS_FILE)
        aspect_labels = torch.load(ASPECT_LABELS_FILE)
        
        logger.info(f"Đã tải dữ liệu. Kích thước:")
        logger.info(f"- Input IDs: {input_ids.shape}")
        logger.info(f"- Attention Masks: {attention_masks.shape}")
        logger.info(f"- Sentiment Labels: {sentiment_labels.shape}")
        logger.info(f"- Aspect Labels: {aspect_labels.shape}")
        
        # Chia tập dữ liệu thành train (80%), val (10%), test (10%)
        indices = torch.randperm(len(input_ids))
        train_size = int(0.8 * len(input_ids))
        val_size = int(0.1 * len(input_ids))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        train_dataset = MultiTaskDataset(
            input_ids=input_ids[train_indices],
            attention_masks=attention_masks[train_indices],
            sentiment_labels=sentiment_labels[train_indices],
            aspect_labels=aspect_labels[train_indices]
        )
        
        val_dataset = MultiTaskDataset(
            input_ids=input_ids[val_indices],
            attention_masks=attention_masks[val_indices],
            sentiment_labels=sentiment_labels[val_indices],
            aspect_labels=aspect_labels[val_indices]
        )
        
        test_dataset = MultiTaskDataset(
            input_ids=input_ids[test_indices],
            attention_masks=attention_masks[test_indices],
            sentiment_labels=sentiment_labels[test_indices],
            aspect_labels=aspect_labels[test_indices]
        )
        
        logger.info(f"Tập train: {len(train_dataset)} mẫu")
        logger.info(f"Tập validation: {len(val_dataset)} mẫu")
        logger.info(f"Tập test: {len(test_dataset)} mẫu")
        
        return train_dataset, val_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"Lỗi khi tải dữ liệu: {e}")
        return None, None, None

def main():
    logger.info("=== BẮT ĐẦU QUÁ TRÌNH FINE-TUNE PHOBERT ===")
    
    # Thiết lập seed
    set_seed(SEED)
    
    # Kiểm tra GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Sử dụng thiết bị: {device}")
    
    # Tải dữ liệu
    train_dataset, val_dataset, test_dataset = load_and_prepare_data()
    
    if train_dataset is None or val_dataset is None or test_dataset is None:
        logger.error("Không thể tải dữ liệu. Hủy quá trình fine-tune.")
        return
    
    # Tạo DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=BATCH_SIZE
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )
    
    # Khởi tạo mô hình
    model = MultiTaskPhoBERTCNN(NUM_SENTIMENT_CLASSES, NUM_ASPECT_CLASSES)
    model.to(device)
    
    # Log cấu trúc mô hình
    logger.info(f"Cấu trúc mô hình: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Tổng số tham số: {total_params:,}")
    logger.info(f"Số tham số có thể huấn luyện: {trainable_params:,}")
    
    # Chuẩn bị optimizer và scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
    
    total_steps = len(train_dataloader) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Fine-tune mô hình
    logger.info(f"Bắt đầu fine-tune với tối đa {MAX_EPOCHS} epochs...")
    model, train_stats, test_metrics = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE
    )
    
    logger.info("Quá trình fine-tune đã hoàn tất!")
    logger.info(f"Mô hình đã được lưu tại: {FINETUNED_MODEL_DIR}")
    logger.info(f"Kết quả đánh giá và biểu đồ đã được lưu tại: {RUN_LOG_DIR}")
    
    # Tóm tắt kết quả
    logger.info("\n=== KẾT QUẢ CUỐI CÙNG ===")
    logger.info(f"Sentiment - Accuracy: {test_metrics['sentiment']['accuracy']:.4f}")
    logger.info(f"Sentiment - F1 Score: {test_metrics['sentiment']['f1_weighted']:.4f}")
    logger.info(f"Aspect - F1 Micro: {test_metrics['aspect']['f1_micro']:.4f}")
    logger.info(f"Aspect - F1 Macro: {test_metrics['aspect']['f1_macro']:.4f}")
    
    logger.info("=== KẾT THÚC QUÁ TRÌNH FINE-TUNE PHOBERT ===")

if __name__ == "__main__":
    main() 