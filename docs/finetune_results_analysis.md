# Phân tích kết quả Fine-tuning PhoBERT cho phân loại Sentiment và Aspect

## 1. Tổng quan

Báo cáo này phân tích kết quả sau quá trình fine-tuning mô hình PhoBERT cho nhiệm vụ phân loại đồng thời cảm xúc (sentiment) và khía cạnh (aspect) từ đánh giá sản phẩm trên Tiki. Mô hình được xây dựng với kiến trúc đa nhiệm vụ (multi-task) kết hợp PhoBERT với CNN để cải thiện hiệu suất trích xuất đặc trưng.

### 1.1. Mô hình sử dụng

- **Mô hình nền tảng:** PhoBERT (pretrained BERT cho tiếng Việt)
- **Kiến trúc:** Multi-task learning kết hợp CNN
- **Nhiệm vụ:**
  - Phân loại cảm xúc: 5 lớp (rất tiêu cực, tiêu cực, trung lập, tích cực, rất tích cực)
  - Phân loại khía cạnh: 5 lớp (other, cskh, quality, price, ship)

### 1.2. Chi tiết kiến trúc CNN kết hợp PhoBERT

Mô hình sử dụng kiến trúc kết hợp giữa PhoBERT và mạng nơ-ron tích chập (CNN) với những đặc điểm sau:

- **PhoBERT làm encoder cơ bản:** Mô hình sử dụng PhoBERT để trích xuất biểu diễn ngữ nghĩa từ văn bản
- **Kiến trúc CNN đa kênh:** Sử dụng nhiều kernel với kích thước khác nhau (3, 4, 5) để bắt các mẫu cục bộ ở các phạm vi khác nhau
- **Max-pooling:** Áp dụng max-over-time pooling sau mỗi lớp tích chập để trích xuất tính năng quan trọng nhất
- **Kết hợp biểu diễn:** Ghép token [CLS] từ PhoBERT với các đặc trưng từ CNN để tạo biểu diễn đa chiều hơn
- **Tất cả các tầng (layers):**
  1. PhoBERT encoder: Chuyển đổi văn bản thành chuỗi embedding có kích thước [batch_size, sequence_length, hidden_size]
  2. CNN Block: Gồm 3 kênh tích chập 1D (kernel sizes: 3, 4, 5) với 128 bộ lọc mỗi kênh
  3. Fully Connected + ReLU: 2 tầng fully-connected (256 neurons) với kích hoạt ReLU
  4. Output Layers: 2 tầng riêng biệt cho phân loại sentiment (5 neurons) và aspect (5 neurons)

Ưu điểm của CNN trong mô hình này:
- Khả năng trích xuất các đặc trưng cục bộ từ chuỗi (n-grams)
- Bắt được các mẫu ngữ cảnh quan trọng không phụ thuộc vào vị trí
- Bổ sung thông tin chéo giữa các từ không liền kề
- Giảm độ phức tạp tính toán so với các mô hình dựa trên RNN/LSTM

### 1.3. Dữ liệu

- **Nguồn dữ liệu:** Đánh giá sản phẩm từ Tiki
- **Phân chia dữ liệu:** Train (80%), Validation (10%), Test (10%)
- **Phân bố dữ liệu test:**
  - Sentiment: 1.352 mẫu (không cân bằng, chủ yếu là "rất tích cực" - 1.035 mẫu)
  - Aspect: Không cân bằng, nhiều nhất là "quality" (1.152) và "ship" (606)

## 2. Kết quả tổng thể

### 2.1. Hiệu suất phân loại cảm xúc (Sentiment)

| Metric | Giá trị |
|--------|---------|
| Accuracy | 79.36% |
| F1 Score (weighted) | 78.99% |
| Precision (weighted) | 78.88% |
| Recall (weighted) | 79.36% |

### 2.2. Hiệu suất phân loại khía cạnh (Aspect)

| Metric | Giá trị |
|--------|---------|
| F1 Score (micro) | 95.83% |
| F1 Score (macro) | 75.64% |
| Precision (micro) | 95.83% |
| Recall (micro) | 95.83% |

## 3. Phân tích chi tiết theo lớp

### 3.1. Phân loại Sentiment

| Lớp | Precision | Recall | F1-score | Support |
|-----|-----------|--------|----------|---------|
| Rất tiêu cực | 0.63 | 0.42 | 0.50 | 62 |
| Tiêu cực | 0.19 | 0.13 | 0.16 | 38 |
| Trung lập | 0.33 | 0.38 | 0.35 | 64 |
| Tích cực | 0.38 | 0.39 | 0.39 | 153 |
| Rất tích cực | 0.91 | 0.93 | 0.92 | 1.035 |

### 3.2. Phân loại Aspect

| Khía cạnh | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| Other | 0.00 | 0.00 | 0.00 | 0 |
| CSKH | 0.97 | 0.93 | 0.95 | 111 |
| Quality | 0.94 | 0.97 | 0.96 | 1.152 |
| Price | 0.94 | 0.83 | 0.88 | 266 |
| Ship | 0.99 | 0.99 | 0.99 | 606 |

## 4. Phân tích và nhận xét

### 4.1. Điểm mạnh

1. **Hiệu suất phân loại aspect rất tốt:**
   - F1-score micro đạt 95.83%, cho thấy mô hình phân loại chính xác các khía cạnh
   - Đặc biệt xuất sắc với khía cạnh "ship" (F1 = 99.34%) và "quality" (F1 = 95.72%)

2. **Phân loại tốt cảm xúc tích cực:**
   - Mô hình nhận diện rất tốt các đánh giá "rất tích cực" (F1 = 91.72%)
   - Độ chính xác và recall cao cho lớp này (precision = 0.91, recall = 0.93)

3. **Kiến trúc CNN cải thiện hiệu suất:**
   - Việc kết hợp CNN với PhoBERT giúp mô hình có khả năng trích xuất đặc trưng tốt hơn
   - Các kernel khác kích thước bắt được các mẫu ngữ nghĩa ở các cấp độ khác nhau (từ, cụm từ, mệnh đề)
   - Max-pooling giúp chọn ra các đặc trưng quan trọng nhất, giảm nhiễu
   - Kết hợp biểu diễn [CLS] từ BERT với đặc trưng CNN tạo biểu diễn phong phú hơn

### 4.2. Điểm yếu

1. **Hiệu suất kém với các lớp sentiment thiểu số:**
   - Lớp "tiêu cực" có F1-score rất thấp (16%), với precision chỉ 0.19 và recall 0.13
   - Các lớp "trung lập" và "tích cực" cũng có hiệu suất thấp (35% và 39%)
   - Phân tích ma trận nhầm lẫn cho thấy mô hình thường phân loại sai các lớp thiểu số thành "rất tích cực"
   - Hiện tượng "overwhelming positive bias": mô hình có xu hướng dự đoán lớp chiếm đa số

2. **Dữ liệu mất cân bằng nghiêm trọng:**
   - Phân bố không đều giữa các lớp sentiment (1.035 mẫu "rất tích cực" chiếm 76.5% tập test)
   - Tỷ lệ "rất tích cực" gấp 27 lần "tiêu cực" (1035 vs 38 mẫu), tạo bias lớn trong quá trình học
   - Khía cạnh "other" không có mẫu nào trong tập test, dẫn đến F1-score = 0
   - Sự chênh lệch giữa các lớp aspect (quality: 1152, ship: 606 vs cskh: 111) ảnh hưởng đến độ tin cậy của các metrics tổng thể

3. **Khoảng cách giữa F1-micro và F1-macro:**
   - F1-micro (95.83%) cao hơn nhiều so với F1-macro (75.64%) cho aspect classification
   - Nguyên nhân: F1-micro có xu hướng bị chi phối bởi các lớp đa số
   - F1-macro thấp hơn nhiều do F1-score = 0 của lớp "other" kéo xuống
   - Chênh lệch này là dấu hiệu của vấn đề mất cân bằng dữ liệu nghiêm trọng

4. **Hạn chế của kiến trúc CNN:**
   - Mặc dù CNN giúp cải thiện hiệu suất, nhưng vẫn có một số hạn chế:
     - CNN không bắt được mối quan hệ xa giữa các từ (long-range dependencies)
     - Khả năng học biểu diễn phân cấp bị hạn chế
     - Có thể bỏ qua một số đặc trưng quan trọng sau bước max-pooling
     - Số lượng kênh và kích thước kernel cố định có thể không tối ưu cho mọi loại văn bản

5. **Tỷ lệ dự đoán sai các trường hợp khó:**
   - Phân tích các lỗi dự đoán cho thấy mô hình gặp khó khăn với:
     - Các đánh giá có nhiều khía cạnh khác nhau
     - Các phát biểu mỉa mai hoặc ẩn ý
     - Phát biểu có cả nội dung tích cực và tiêu cực
     - Các đánh giá sử dụng từ ngữ địa phương hoặc từ lóng

## 5. Đề xuất cải tiến chi tiết

1. **Giải pháp cụ thể cho vấn đề mất cân bằng dữ liệu:**
   - **Class weighting cụ thể:** Áp dụng trọng số ngược với tần suất xuất hiện của từng lớp trong loss function:
     ```python
     class_weights = {0: 21.8, 1: 35.6, 2: 21.1, 3: 8.8, 4: 1.3}  # Ví dụ trọng số
     sentiment_criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
     ```
   - **Focal Loss:** Thay thế CrossEntropy bằng Focal Loss để tập trung vào các mẫu khó phân loại:
     ```python
     # gamma > 0 làm giảm ảnh hưởng của các mẫu dễ phân loại
     def focal_loss(predictions, targets, alpha=0.25, gamma=2.0):
         # implementation code
     ```
   - **Mixed Sampling:** Kết hợp oversampling các lớp thiểu số và undersampling lớp đa số

2. **Cải thiện kiến trúc CNN:**
   - **CNN kết hợp Attention:** Thêm cơ chế attention để tập trung vào các phần quan trọng của chuỗi:
     ```python
     class AttentionLayer(nn.Module):
         def __init__(self, hidden_size):
             super(AttentionLayer, self).__init__()
             self.attention = nn.Linear(hidden_size, 1)
             
         def forward(self, x):
             # x: batch_size x seq_length x hidden_size
             attention_weights = torch.softmax(self.attention(x), dim=1)
             context_vector = torch.sum(attention_weights * x, dim=1)
             return context_vector, attention_weights
     ```
   - **Dynamic CNN:** Sử dụng kiến trúc CNN động với k-max pooling thay vì chỉ max pooling:
     ```python
     def k_max_pooling(x, dim, k):
         index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
         return x.gather(dim, index)
     ```
   - **Dilated CNN:** Sử dụng tích chập giãn (dilated convolution) để bắt các mối quan hệ xa mà không tăng số lượng tham số:
     ```python
     nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=2)
     ```

3. **Kỹ thuật ensemble nâng cao:**
   - **Stacking ensemble:** Huấn luyện một meta-model để kết hợp dự đoán từ nhiều mô hình cơ sở:
     1. PhoBERT + CNN (mô hình hiện tại)
     2. PhoBERT + BiLSTM
     3. XLM-RoBERTa + CNN
     4. VinAI/BERTweet (BERT được pre-train trên dữ liệu Twitter)
   - **Bagging:** Train nhiều mô hình trên các tập dữ liệu bootstrap khác nhau:
     ```python
     def bootstrap_sample(X, y, n_samples=None):
         n_samples = len(X) if n_samples is None else n_samples
         idx = np.random.choice(len(X), size=n_samples, replace=True)
         return X[idx], y[idx]
     ```
   - **Task-specific Heads:** Huấn luyện các đầu ra chuyên biệt cho từng lớp sentiment và aspect:
     ```python
     class MultiHeadModel(nn.Module):
         def __init__(self):
             # Mô hình chung
             self.shared_encoder = SharedEncoder()
             # Đầu ra riêng cho từng lớp
             self.sentiment_heads = nn.ModuleList([BinaryClassifier() for _ in range(5)])
             self.aspect_heads = nn.ModuleList([BinaryClassifier() for _ in range(5)])
     ```

4. **Chiến lược thu thập và xử lý dữ liệu:**
   - **Data augmentation cho tiếng Việt:** Tạo dữ liệu mới cho các lớp thiểu số:
     - Phương pháp EDA (Easy Data Augmentation): Thay thế từ đồng nghĩa, hoán vị từ ngẫu nhiên
     - Back-translation: Dịch sang ngôn ngữ khác rồi dịch ngược lại
     - Sử dụng mô hình ngôn ngữ để tạo dữ liệu tổng hợp
   - **Hard negative mining:** Tập trung thu thập các mẫu khó (dễ bị phân loại sai):
     1. Train mô hình ban đầu
     2. Dự đoán trên tập dữ liệu chưa gán nhãn
     3. Thu thập các mẫu mô hình không chắc chắn (low confidence)
     4. Gán nhãn thủ công và thêm vào tập huấn luyện
   - **Active learning pipeline:** Xây dựng quy trình gán nhãn hiệu quả:
     ```python
     def uncertainty_sampling(model, unlabeled_data, k):
         probs = model.predict_proba(unlabeled_data)
         uncertainties = 1 - np.max(probs, axis=1)
         return np.argsort(uncertainties)[-k:]  # Top-k uncertain samples
     ```

5. **Tối ưu hóa và regularization nâng cao:**
   - **Learning rate scheduling:** Lịch trình learning rate thích ứng:
     ```python
     scheduler = torch.optim.lr_scheduler.OneCycleLR(
         optimizer, max_lr=2e-5, total_steps=total_steps,
         pct_start=0.1, anneal_strategy='cos'
     )
     ```
   - **Mixup cho NLP:** Áp dụng kỹ thuật mixup cho dữ liệu văn bản:
     ```python
     def mixup(x1, x2, y1, y2, alpha=0.2):
         lambda_ = np.random.beta(alpha, alpha)
         mixed_x = lambda_ * x1 + (1 - lambda_) * x2
         mixed_y = lambda_ * y1 + (1 - lambda_) * y2
         return mixed_x, mixed_y
     ```
   - **Gradient accumulation:** Cho phép sử dụng batch size lớn hơn trên phần cứng hạn chế:
     ```python
     # Accumulate gradients over n steps
     accumulation_steps = 4
     for i, batch in enumerate(train_dataloader):
         loss = model(batch)
         loss = loss / accumulation_steps
         loss.backward()
         if (i + 1) % accumulation_steps == 0:
             optimizer.step()
             optimizer.zero_grad()
     ```
   - **Hiệu chỉnh phức tạp mô hình:** Áp dụng kỹ thuật Knowledge Distillation để giảm kích thước mô hình:
     ```python
     def knowledge_distillation_loss(student_logits, teacher_logits, targets, T=2.0, alpha=0.5):
         hard_loss = F.cross_entropy(student_logits, targets)
         soft_targets = F.softmax(teacher_logits / T, dim=1)
         soft_prob = F.log_softmax(student_logits / T, dim=1)
         soft_loss = -torch.sum(soft_targets * soft_prob) / student_logits.size(0)
         return alpha * hard_loss + (1 - alpha) * (T ** 2) * soft_loss
     ```

## 6. Kết luận

Mô hình PhoBERT kết hợp CNN cho thấy tiềm năng lớn trong nhiệm vụ phân loại đồng thời sentiment và aspect cho đánh giá sản phẩm tiếng Việt. Hiệu suất tổng thể khá tốt (accuracy 79.36% cho sentiment, F1-micro 95.83% cho aspect), nhưng vẫn còn nhiều cơ hội để cải thiện, đặc biệt là với các lớp thiểu số.

Các phân tích chi tiết về CNN, điểm mạnh và điểm yếu của mô hình đã cho thấy những hướng tiếp cận cụ thể để nâng cao hiệu suất. Đặc biệt, vấn đề mất cân bằng dữ liệu và hiệu suất kém với các lớp thiểu số cần được ưu tiên giải quyết thông qua các kỹ thuật sampling, loss function đặc biệt và cải tiến kiến trúc mô hình.

Kết quả cho thấy việc kết hợp transfer learning (sử dụng mô hình pretrained) với mạng CNN đặc thù là hướng đi hiệu quả cho bài toán phân tích cảm xúc đa nhiệm vụ trong tiếng Việt. Các đề xuất cải tiến nêu trên dự kiến sẽ giúp nâng cao hiệu suất tổng thể của mô hình, đặc biệt là với các lớp thiểu số, và tạo nền tảng vững chắc cho các ứng dụng phân tích cảm xúc tiếng Việt trong thương mại điện tử. 