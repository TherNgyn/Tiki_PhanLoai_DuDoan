import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Tiki Product Analysis",
    page_icon="📊",
    layout="wide"
)

# Hàm xử lý văn bản
def preprocess_text(text):
    if isinstance(text, str):
        # Chuyển văn bản về chữ thường
        text = text.lower()
        
        # Loại bỏ dấu câu và các ký tự đặc biệt
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Loại bỏ chữ số
        text = re.sub(r'\d+', ' ', text)
        
        # Loại bỏ các khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    return ""

# Tải các mô hình và dữ liệu đã được lưu
@st.cache_resource
def load_models():
    # Mô hình phân loại
    classifier = joblib.load('product_category_model.pkl') 
    tfidf_naiye = joblib.load('tfidf_naiye.pkl')
    # Mô hình dự đoán lượng bán
    sales_model = joblib.load('quantity_sold_prediction_model.pkl')
    sales_scaler = joblib.load('sales_scaler.pkl')
    sales_tfidf = joblib.load('sales_tfidf.pkl')
    sales_features = joblib.load('sales_features.pkl')

    # Danh mục 
    categories = joblib.load('category_mapping.pkl')
    
    return {
        'classifier': classifier,
        'tfidf_naiye': tfidf_naiye,
        'sales_model': sales_model,
        'sales_scaler': sales_scaler,
        'sales_tfidf': sales_tfidf,
        'sales_features': sales_features,
        'categories': categories
    }
try:
    models = load_models()
except:
    st.stop()

# Tiêu đề ứng dụng
st.title("Tiki Product Analytics")
st.markdown("### Phân tích, dự đoán và đề xuất sản phẩm")
# Thêm màu sắc cho tiêu đề

# Tab cho các chức năng khác nhau
tab1, tab2 = st.tabs(["Phân loại sản phẩm", "Dự đoán lượng bán"])

# Tab 1: Phân loại sản phẩm
with tab1:
    st.header("Phân loại sản phẩm")
    # Nhập thông tin sản phẩm
    product_name = st.text_input("Nhập tên sản phẩm:", "Sách Digital Marketing")
    if st.button("Phân loại"):
        if product_name:
            # Tiền xử lý tên sản phẩm
            processed_name = preprocess_text(product_name)
            # Chuyển đổi tên sản phẩm thành vector TF-IDF
            tfidf_vector = models['tfidf_naiye'].transform([processed_name]).toarray()
            # Dự đoán danh mục
            prediction = models['classifier'].predict(tfidf_vector)[0]
            probabilities = models['classifier'].predict_proba(tfidf_vector)[0]
            
            # Hiển thị kết quả
            st.success(f"Danh mục dự đoán: **{prediction}**")
            
            # Hiển thị top 3 danh mục có xác suất cao nhất
            categories = models['classifier'].classes_
            top3_indices = probabilities.argsort()[-3:][::-1]
            top3_categories = categories[top3_indices]
            top3_probabilities = probabilities[top3_indices]
            
            # Vẽ biểu đồ
            fig, ax = plt.subplots()
            # Dùng màu cmap
            colors = plt.cm.Greens(np.linspace(1, 0.5, len(top3_categories)))
            bars = ax.bar(top3_categories, top3_probabilities * 100, color=colors)
            ax.set_ylabel('Xác suất (%)')
            ax.set_xlabel('Danh mục')
            # In nghiêng tên danh mục ở trục x
            ax.set_xticklabels(top3_categories, rotation=45, ha='right', fontsize=10, fontstyle='italic')
            ax.set_title('Top 3 danh mục sản phẩm có xác suất thuộc về cao nhất')
            st.pyplot(fig)
        else:
            st.warning("Vui lòng nhập tên sản phẩm.")

# Tab 2: Dự đoán lượng bán
with tab2:
    st.header("Dự đoán lượng bán")
    
    # Form nhập thông tin sản phẩm
    col1, col2 = st.columns(2)
    
    # Lấy ánh xạ danh mục
    category_mapping = models['categories']
    
    # Tạo ánh xạ ngược để lấy mã danh mục từ tên hiển thị
    reverse_category_mapping = {v: k for k, v in category_mapping.items()}
    
    # Lấy danh sách tên hiển thị của các danh mục đã được sắp xếp
    display_categories = sorted(category_mapping.values())
    
    with col1:
        name = st.text_input("Tên sản phẩm:", "Sách Digital Marketing")
        price = st.number_input("Giá (VNĐ):", min_value=1000, value=150000, step=10000)
        list_price = st.number_input("Giá niêm yết (VNĐ):", min_value=1000, value=200000, step=10000)
    
    with col2:
        # Sử dụng tên hiển thị thân thiện với người dùng
        selected_category_name = st.selectbox("Danh mục:", display_categories)
        
        # Chuyển đổi lại thành mã danh mục cho xử lý bên trong
        selected_category_code = reverse_category_mapping[selected_category_name]
        
        review_count = st.number_input("Số lượng đánh giá (Reviews) hiện tại:", min_value=0, value=10, step=1)
        has_discount = st.checkbox("Có giảm giá")
    
    # Tính discount và discount_rate nếu có giảm giá
    if has_discount:
        discount = list_price - price
        discount_rate = int((discount/list_price) * 100)
    else:
        discount = 0
        discount_rate = 0
    
    if st.button("Dự đoán lượng bán"):
            # Tiền xử lý dữ liệu
            processed_name = preprocess_text(name)
            
            # Mã hóa danh mục - sử dụng category_encoder được định nghĩa trước đó hoặc giá trị mặc định
            # Nếu có category_encoder cụ thể, hãy sử dụng nó
            try:
                category_encoder = models.get('category_encoder', {})
                category_encoded = category_encoder.get(selected_category_code, 0)
            except:
                # Fallback đơn giản nếu không có encoder
                category_encoded = {
                    'nha-sach-tiki': 0,
                    'dien-thoai-may-tinh-bang': 1,
                    'thiet-bi-kts-phu-kien-so': 2
                }.get(selected_category_code, 0)
            
            # Chuẩn bị dữ liệu đầu vào
            numeric_input = np.array([[price, list_price, discount, discount_rate, review_count, has_discount, category_encoded]])
            
            # Chuyển đổi tên sản phẩm thành vector TF-IDF
            tfidf_vector = models['sales_tfidf'].transform([processed_name]).toarray()
            
            # Kết hợp dữ liệu
            X_input = np.hstack([numeric_input, tfidf_vector])
            
            # Chuẩn hóa dữ liệu
            X_scaled = models['sales_scaler'].transform(X_input)
            
            # Dự đoán
            log_prediction = models['sales_model'].predict(X_scaled)[0]
            prediction = np.expm1(log_prediction)  # Chuyển đổi từ log scale
            
            # Hiển thị kết quả
            st.success(f"Lượng bán dự kiến: **{int(prediction):,} sản phẩm**")

# Thêm icon 
st.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.image("https://salt.tikicdn.com/ts/upload/0e/07/78/ee828743c9afa9792cf20d75995e134e.png", width=100)
st.markdown("Phát triển bởi Nhóm 4 | Dữ liệu từ Tiki.vn")