import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(filepath="data/data.csv"):
    """
    Loads raw data, performs feature engineering, removes outliers, 
    and splits into train/test sets.
    """
    df = pd.read_csv(filepath)
    
    columns_to_drop = ['date', 'street', 'statezip', 'country']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    if 'city' in df.columns:
        df['city'] = LabelEncoder().fit_transform(df['city'])
        
    # 4. Remove outliers (prices that are 0 or extremely high)
    df = df[(df['price'] > 0) & (df['price'] < 2_000_000)].copy()
    

    df['house_age'] = 2026 - df['yr_built']
    
    df['is_renovated'] = (df['yr_renovated'] > 0).astype(int)
    
    df['total_area'] = df['sqft_living'] + df['sqft_lot']
    # --------------------------------
    
    X = df.drop(columns=['price'])
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare_data("../data/data.csv")
    print(f"Data prep successful! Train set shape with new features: {X_train.shape}")