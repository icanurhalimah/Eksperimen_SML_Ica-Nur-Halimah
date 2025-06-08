import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_dataset(input_path="../personality_dataset.csv"):
    # Load dataset
    df = pd.read_csv(input_path)

    # Drop missing values
    df.dropna(inplace=True)

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Pisahkan fitur dan label
    X = df.drop('Personality', axis=1)
    y = df['Personality']

    # Copy data untuk diproses
    X_processed = X.copy()
    y_processed = y.copy()

    # Standarisasi fitur numerik
    numerical_cols = X.select_dtypes(include='float').columns
    scaler = StandardScaler()
    X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])
    X_processed['Stage_fear'] = X_processed['Stage_fear'].map({'Yes': 1, 'No': 0})
    X_processed['Drained_after_socializing'] = X_processed['Drained_after_socializing'].map({'Yes': 1, 'No': 0})

    # Label encoding untuk target
    le = LabelEncoder()
    y_processed = le.fit_transform(y_processed)

    # Gabungkan kembali
    data_preprocessed = X_processed.copy()
    data_preprocessed['Personality'] = y_processed

    # Simpan ke file CSV
    output_file = 'personality_dataset_clean.csv'
    scaler_file = 'scaler.pkl'
    le_file = 'le.pkl'
    data_preprocessed.to_csv(output_file, index=False)
    print(f"Data yang sudah diproses telah disimpan sebagai '{output_file}'")
    joblib.dump(scaler, scaler_file)
    print(f"StandardScaler telah disimpan sebagai '{scaler_file}'")
    joblib.dump(le, le_file)
    print(f"Label Encoder telah disimpan sebagai '{le_file}'")
    return data_preprocessed

# Jalankan langsung saat file dipanggil
if __name__ == "__main__":
    preprocess_dataset()