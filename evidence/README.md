# Phân Tích Đánh Giá RAGAS: Prompt V1 vs Prompt V2

Tài liệu phân tích kết quả thử nghiệm và đánh giá chất lượng RAG pipeline giữa 2 phiên bản prompt: **Prompt V1** (`namnt1204-day22-rag-v1`) và **Prompt V2** (`namnt1204-day22-rag-v2`) trên tập 50 câu hỏi benchmark (nguồn: `evidence/03_ragas_report.json`).

---

## 1. Bảng So Sánh Số Liệu Thực Tế

| RAGAS Metric | Prompt V1 | Prompt V2 | Chênh lệch (V1 - V2) | Kết quả / Winner |
|:---|:---:|:---:|:---:|:---:|
| **Faithfulness** | **0.9927** | 0.9737 | +0.0190 | 🏆 **Prompt V1** |
| **Answer Relevancy** | **0.8492** | 0.8418 | +0.0074 | 🏆 **Prompt V1** |
| **Context Recall** | **0.9800** | **0.9800** | 0.0000 | 🤝 **HÒA (TIE)** |
| **Context Precision** | **0.9633** | **0.9633** | 0.0000 | 🤝 **HÒA (TIE)** |

*Ghi chú: Mục tiêu đề bài đặt ra là `faithfulness >= 0.8`.*

---

## 2. Phân Tích Chi Tiết Từng Chỉ Số

### 2.1. Faithfulness (Độ trung thực với ngữ cảnh)
- **V1: 0.9927 vs V2: 0.9737 (V1 vượt trội +1.9%)**
- **Nguyên nhân**: Prompt V1 định nghĩa chỉ thị rõ ràng và khắt khe hơn về việc chỉ sử dụng thông tin có sẵn trong retrieved context, không suy diễn hoặc bổ sung kiến thức bên ngoài nếu không có căn cứ. Prompt V2 tuy diễn đạt tự nhiên hơn nhưng đôi khi có xu hướng khái quát hóa câu trả lời, dẫn đến việc evaluator phát hiện một số phát biểu không hoàn toàn bám sát nguyên văn ngữ cảnh.
- Cả hai phiên bản đều vượt xa ngưỡng mục tiêu tối thiểu (0.8), khẳng định khả năng kiểm soát hallucination của toàn bộ hệ thống là cực kỳ tốt.

### 2.2. Answer Relevancy (Độ phù hợp của câu trả lời)
- **V1: 0.8492 vs V2: 0.8418 (V1 cao hơn nhẹ +0.7%)**
- **Nguyên nhân**: Prompt V1 tập trung trả lời trực diện vào câu hỏi trọng tâm, giúp câu trả lời cô đọng và khớp chính xác với ý định người dùng (User Intent). Prompt V2 có phong cách giải thích mở rộng hơn, đôi lúc chứa các thông tin bổ trợ không trực tiếp thuộc về câu hỏi chính.

### 2.3. Context Recall & Context Precision (Chỉ số của tầng Retrieval)
- **V1 = V2 = 0.9800 (Recall) và 0.9633 (Precision)**
- **Nguyên nhân**: Hai chỉ số này đo lường hiệu năng của tầng truy xuất (Retrieval Layer). Do cả hai thử nghiệm V1 và V2 đều sử dụng chung vector database FAISS và cùng mô hình embedding (`text-embedding-004`) với cùng tham số `k=3`, chất lượng các đoạn context được trả về là hoàn toàn đồng nhất.

---

## 3. Kết Luận & Khuyến Nghị Triển Khai Production

1. **Khuyến nghị**: **Chọn Prompt V1 làm phiên bản chính thức (Default Route)** cho production.
2. **Lý do**:
   - Vượt trội ở cả 2 chỉ số sinh câu trả lời quan trọng nhất: **Faithfulness (0.9927)** và **Answer Relevancy (0.8492)**.
   - Giảm thiểu tối đa nguy cơ hallucination trong các tác vụ truy vấn tài liệu kỹ thuật nhạy cảm.
3. **Chiến lược A/B Testing tiếp theo**:
   - Đặt tỉ lệ routing 90% traffic cho V1 và 10% canary traffic cho các biến thể prompt thử nghiệm mới trong tương lai.
