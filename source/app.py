import streamlit as st
import pandas as pd
import joblib
import numpy as np


st.set_page_config(page_title="Dự báo Nguy cơ Bệnh tim", layout="wide")

st.title("🩺 Hệ thống Dự báo Sức khỏe")
st.write("Nhập thông số bệnh nhân để dự báo khả năng mắc bệnh.")

@st.cache_resource # Dùng cache để không phải load lại mô hình mỗi khi nhấn nút
def load_model():
    return joblib.load('heart_disease.pkl')

data = load_model()
model = data['model']
features = data['all_cols']

# tạo sidebar
st.sidebar.header("Nhập các chỉ số tại đây")

def user_input_features():
    # NHÓM BIẾN SỐ (NUMERICAL)
    age = st.sidebar.slider('Tuổi', 18, 100, 50)
    trestbps = st.sidebar.slider('Huyết áp lúc nghỉ (mm Hg)', 80, 200, 120)
    chol = st.sidebar.slider('Nồng độ Cholesterol (mg/dl)', 100, 600, 200)
    thalach = st.sidebar.number_input('Nhịp tim tối đa (thalach)', 60, 220, 150)
    oldpeak = st.sidebar.slider('ST depression (oldpeak)', 0.0, 6.0, 1.0)
    
    # NHÓM BIẾN PHÂN LOẠI (CATEGORICAL)
    sex = st.sidebar.selectbox('Giới tính', options=[(1, "Nam"), (0, "Nữ")], format_func=lambda x: x[1])[0]
    
    fbs = st.sidebar.selectbox('Đường huyết lúc đói > 120 mg/dl', options=[(1, "Có (1)"), (0, "Không (0)")], format_func=lambda x: x[1])[0]
    
    exang = st.sidebar.selectbox('Đau ngực khi tập thể dục', options=[(1, "Có (1)"), (0, "Không (0)")], format_func=lambda x: x[1])[0]
    
    slope = st.sidebar.selectbox('Độ dốc đoạn ST (slope)', options=[(1, "Lên (1)"), (2, "Đi ngang (2)"), (3, "Xuống (3)")], format_func=lambda x: x[1])[0]
    
    ca = st.sidebar.selectbox('Số mạch máu chính (ca)', options=[0, 1, 2, 3])

    cp = st.sidebar.selectbox('Loại đau ngực (cp)', 
                               options=[(1, "Điển hình (1)"), (2, "Không điển hình (2)"), (3, "Không đau ngực (3)"), (4, "Không triệu chứng (4)")],
                               format_func=lambda x: x[1])[0]
    
    restecg = st.sidebar.selectbox('Kết quả điện tâm đồ (restecg)',
                                    options=[(0, "Bình thường (0)"), (1, "Sóng ST-T bất thường (1)"), (2, "Phì đại thất trái (2)")],
                                    format_func=lambda x: x[1])[0]

    thal = st.sidebar.selectbox('Thalassemia (thal)',
                                 options=[(3, "Bình thường (3)"), (6, "Khiếm khuyết cố định (6)"), (7, "Khiếm khuyết có thể đảo ngược (7)")],
                                 format_func=lambda x: x[1])[0]
    
    
    # Tạo dictionary chứa dữ liệu nhập

    display_dict = {
        'age': age, 'trestbps': trestbps, 'chol': chol, 'thalach': thalach, 'oldpeak': oldpeak,
        'sex': sex, 'fbs': fbs, 'exang': exang, 'slope': slope, 'ca': ca,
        'cp': cp, 'restecg': restecg, 'thal': thal
    }

    return pd.DataFrame([display_dict])

input_df = user_input_features()

# 4. Hiển thị dữ liệu đã nhập
st.subheader("📋 Thông tin bệnh nhân:")
st.write(input_df)

def transfer(df):
    row = df.iloc[0]
    data_dict = {
        'age': row['age'],
        'trestbps': row['trestbps'],
        'chol': row['chol'],
        'thalach': row['thalach'],
        'oldpeak': row['oldpeak'],
        
        'sex': row['sex'],
        'fbs': row['fbs'],
        'exang': row['exang'],
        
        'slope': row['slope'],
        'ca': row['ca'],
        
        'cp_2': 0,
        'cp_3': 0,
        'cp_4': 0, 
        'restecg_1': 0,
        'restecg_2': 0,
        'thal_1': 0,
        'thal_2': 0
    }
    
    if  row['cp'] == 2: data_dict['cp_2'] = 1
    elif  row['cp'] == 3: data_dict['cp_3'] = 1
    elif  row['cp'] == 4: data_dict['cp_4'] = 1

    if  row['restecg'] == 1: data_dict['restecg_1'] = 1
    elif  row['restecg'] == 2: data_dict['restecg_2'] = 1

    if  row['thal'] == 6: data_dict['thal_1'] = 1
    elif  row['thal'] == 7: data_dict['thal_2'] = 1
    
    return pd.DataFrame([data_dict])

if st.button('Dự báo'):
    model_input = transfer(input_df)
    
    try:
        if 'iqr_bounds' in data:
            for col, lim in data['iqr_bounds'].items():
                if col in model_input.columns:
                    model_input[col] = model_input[col].clip(lower=lim['lower'], upper=lim['upper'])
        
        if 'scaler' in data:
            scale_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            model_input[scale_cols] = data['scaler'].transform(model_input[scale_cols])
            
        final_input = model_input[features]
        
        prediction = model.predict(final_input)[0]
        prediction_proba = model.predict_proba(final_input)[0, 1]

        st.subheader("🩺 Kết quả:")
        if prediction == 1:
            st.error(f"⚠️ Cảnh báo: Có nguy cơ mắc bệnh tim cao! (Xác suất: {prediction_proba:.4%})")
        else:
            st.success(f"✅ An toàn: Nguy cơ mắc bệnh thấp. (Xác suất: {prediction_proba:.4%})")
        
        st.progress(prediction_proba)
        
    except KeyError as e:
        st.warning(f"Danh sách biến trong Model không khớp. Lỗi thiếu cột: {e}")