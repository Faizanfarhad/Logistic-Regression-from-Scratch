from sklearn.preprocessing import LabelEncoder
import pandas as pd

classification_encoder = LabelEncoder()

def encode_categorical_columns(df:pd.DataFrame):
    try:
        ''' return a seprate dataframe for not overwritting '''
        label_encoder = {}
        for col in df.select_dtypes(include=['object', 'string']).columns:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            label_encoder[col] = encoder
            print("Label encode : ",label_encoder)
        return df,label_encoder
    except Exception as e:
        print(f"Error in encode_categorical_columns :  {e}")

def feature_engineering(df:pd.DataFrame):
    cleaned_df = df.copy()
    cleaned_df.drop_duplicates(inplace=True)
    cleaned_df.dropna(inplace=True)
    
    for col in cleaned_df:
        try:
            cleaned_df[col] = cleaned_df[col].astype(str).str.replace(',','').str.replace(' ', '')
            if cleaned_df[col] == cleaned_df[col].str.isnumeric().all():
                cleaned_df[col] = pd.to_numeric(cleaned_df[col])
                cleaned_df[col] = cleaned_df[col].astype('Int64')
        except Exception as e:
            print(f'Errror cannot able to convert string to numeric : {e}')
    return cleaned_df
