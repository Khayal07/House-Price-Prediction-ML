import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(filepath="data/data.csv"):
    """
    Loads raw data, cleans it, removes outliers, and splits into train/test sets.
    """
    
    df = pd.read_csv(filepath)
    
    columns_to_drop = ['date', 'street', 'statezip', 'country']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    if 'city' in df.columns:
        df['city'] = LabelEncoder().fit_transform(df['city'])
        
    df = df[(df['price'] > 0) & (df['price'] < 2_000_000)]
    
    X = df.drop(columns=['price'])
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data("../data/data.csv")
    print(f"Data prep successful! Train set shape: {X_train.shape}")
    