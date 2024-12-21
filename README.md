
# Mô hình Skip-gram cho Tập dữ liệu Tiếng Việt

## Giới thiệu
Kho mã nguồn này chứa việc triển khai và huấn luyện mô hình **Skip-gram** trên một tập dữ liệu văn bản tiếng Việt. Dự án được thực hiện trong khuôn khổ môn học **Xử lý ngôn ngữ tự nhiên** tại Đại học Sài Gòn.

Mô hình Skip-gram là một mạng nơ-ron được thiết kế để học các vector từ vựng (word embeddings), từ đó phát hiện mối quan hệ ngữ nghĩa giữa các từ. Triển khai này tập trung vào tiếng Việt, tận dụng dữ liệu văn bản đa dạng để cải thiện khả năng tổng quát của mô hình.

## Tính năng
- Triển khai mô hình Skip-gram để học từ vựng (word embedding).
- Tiền xử lý văn bản tiếng Việt với các công cụ như [PyVi](https://pypi.org/project/pyvi/).
- Đánh giá các vector từ vựng bằng cách sử dụng độ tương đồng cosine cho các cặp từ đồng nghĩa và trái nghĩa.
- Bao gồm các thí nghiệm so sánh mô hình với các tốc độ học khác nhau.

## Bộ dữ liệu
- Dữ liệu huấn luyện được lấy từ một kho mã nguồn công khai trên GitHub: [news-corpus](https://github.com/binhvq/news-corpus).
- Bao gồm hơn 14 triệu bài báo từ nhiều nguồn tin tức tiếng Việt, đảm bảo sự đa dạng và thực tế trong việc đại diện ngôn ngữ.
- Sau khi tiền xử lý:
  - Số câu: 1000
  - Kích thước từ vựng: 2651
  - Tổng số cặp từ: 23,460

## Các bước tiền xử lý
1. Chuyển văn bản về chữ thường để đảm bảo tính nhất quán.
2. Loại bỏ dấu câu và ký tự đặc biệt.
3. Tách từ sử dụng thư viện **ViTokenizer**.
4. Loại bỏ các từ dừng (stopwords) dựa trên danh sách từ dừng tiếng Việt đã được định nghĩa sẵn.
5. Chuẩn hóa cấu trúc dữ liệu cho đầu vào mô hình.

## Chi tiết mô hình
- **Kiến trúc**:
  - Lớp đầu vào: Các vector từ được mã hóa dạng one-hot.
  - Lớp ẩn: Lớp embedding để học các đại diện từ vựng.
  - Lớp đầu ra: Softmax để dự đoán từ ngữ cảnh.
- **Cấu hình huấn luyện**:
  - Kích thước embedding: 100
  - Kích thước batch: 1
  - Số epoch: 100
  - Tốc độ học: 0.01 (Mô hình V1), 0.05 (Mô hình V2)
  - Hàm mất mát: Cross-Entropy

## Kết quả
- Mô hình V1 (tốc độ học 0.01) cho các vector từ tốt hơn, đặc biệt là với các cặp từ đồng nghĩa.
- Phân tích độ tương đồng cosine cho thấy độ chính xác ngữ nghĩa cao hơn với Mô hình V1 so với Mô hình V2.
- Kết quả ví dụ:
  - Cặp từ đồng nghĩa "học" ↔ "giáo_dục": Độ tương đồng cao.
  - Cặp từ trái nghĩa "tăng" ↔ "giảm": Sự khác biệt ngữ nghĩa rõ ràng.

## Thách thức và bài học kinh nghiệm
- Xử lý văn bản tiếng Việt nhiễu từ các nguồn khác nhau.
- Lựa chọn các siêu tham số tối ưu cho việc huấn luyện hiệu quả.
- Đánh giá các vector từ với tài nguyên tính toán hạn chế.

## Cách sử dụng
### Yêu cầu hệ thống
- Python 3.8 hoặc cao hơn
- Các thư viện yêu cầu: `pyvi`, `numpy`, `scipy`

### Chạy mô hình
1. Clone kho mã nguồn:
   ```bash
   git clone https://github.com/tinbalon3/Skip-grams_not_use_library.git
   cd Skip-grams_not_use_library
   ```

2. Huấn luyện mô hình:
   ```python
   python train.py
   ```

3. Đánh giá các vector từ:
   ```python
   python evaluate.py
   ```

## Tài liệu tham khảo
1. Mikolov, T., et al. (2013). [Các đại diện phân tán của từ và cụm từ](https://arxiv.org/abs/1310.4546).
2. Rong, X. (2014). [Giải thích việc học các tham số của Word2vec](https://arxiv.org/abs/1411.2738).
3. [Tài liệu PyVi](https://pypi.org/project/pyvi/).
4. Binh, V. Q. (chưa có ngày). [news-corpus trên GitHub](https://github.com/binhvq/news-corpus).
5. [Các từ dừng tiếng Việt](https://github.com/stopwords/vietnamese-stopwords).

## Tác giả
- **Dương Văn Sìnl**  
  Sinh viên tại Đại học Sài Gòn, Khoa Công nghệ Thông tin  
  Liên hệ: tinbalon3@gmail.com
```

