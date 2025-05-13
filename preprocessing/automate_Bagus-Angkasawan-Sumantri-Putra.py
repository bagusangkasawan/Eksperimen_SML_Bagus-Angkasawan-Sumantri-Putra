import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load dataset dari file path"""
    return pd.read_csv(file_path)

def detect_outliers_iqr(data):
    """Deteksi outlier menggunakan metode IQR"""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)].index.tolist()
    return outliers

def preprocess_data(df):
    """Melakukan preprocessing data (menghapus missing value, duplikasi, outlier, dll.)"""
    
    # Cek dan hapus missing value
    if df.isnull().sum().any():
        df.dropna(inplace=True)
        print("Missing values dihapus.")
    
    # Cek dan hapus duplikasi
    if df.duplicated().sum() > 0:
        df.drop_duplicates(inplace=True)
        print("Duplikasi dihapus.")
    
    # Deteksi dan hapus outliers
    outliers_age = detect_outliers_iqr(df['Age'])
    outliers_height = detect_outliers_iqr(df['Height'])
    outliers_weight = detect_outliers_iqr(df['Weight'])
    outliers_bmi = detect_outliers_iqr(df['BMI'])
    
    all_outliers = list(set(outliers_age + outliers_height + outliers_weight + outliers_bmi))
    print(f"Jumlah outlier yang terdeteksi: {len(all_outliers)}")
    df.drop(index=all_outliers, inplace=True)
    
    # Label Encoding untuk Gender dan ObesityCategory
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    
    le_target = LabelEncoder()
    df['ObesityCategory'] = le_target.fit_transform(df['ObesityCategory'])
    
    return df

def scale_data(X):
    """Skalakan data menggunakan StandardScaler"""
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def split_data(X, y):
    """Split data menjadi training dan testing"""
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def save_preprocessed_data(df, output_path):
    """Simpan dataset yang sudah diproses"""
    df.to_csv(output_path, index=False)
    print(f"Data yang sudah diproses disimpan di {output_path}")

if __name__ == "__main__":
    # Path dataset mentah (dari folder namadataset_raw)
    raw_data_path = './obesity_data_raw.csv'
    
    # Load dataset
    df = load_data(raw_data_path)
    
    # Proses data
    df = preprocess_data(df)
    
    # Pisahkan fitur dan target
    X = df.drop('ObesityCategory', axis=1)
    y = df['ObesityCategory']
    
    # Scaling data
    X_scaled = scale_data(X)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # Simpan data yang sudah diproses (ke folder namadataset_preprocessing)
    processed_data_path = '../preprocessing/obesity_data_preprocessing.csv' 
    save_preprocessed_data(df, processed_data_path)
