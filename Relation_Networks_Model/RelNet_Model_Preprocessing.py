# -*- coding: utf-8 -*-
"""
Written by Kambana on February 2nd 2026
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder

def encode_proto_features(df):
    # Check if the 'proto' column exists
    if 'proto' in df.columns:
        # Apply the encoding rules
        df['proto'] = df['proto'].apply(lambda x: 1 if x == 'TCP' 
                                        else 0 if x == 'UDP' 
                                        else 2 if x == 'ICMP' 
                                        else 3)
    return df

def impute_numeric_columns(df):
    """
    Clean and fill int64 and float32 columns:
    - Fill int64 (and hash trick) with 0
    - Fill float32 with the median
    """

    # Find columns by type
    int_cols = df.select_dtypes(include=["int64"]).columns
    float_cols = df.select_dtypes(include=["float32"]).columns

    # Fill int64 columns (including hash trick)
    if len(int_cols) > 0:
        imputer_int = SimpleImputer(strategy="constant", fill_value=0)
        df[int_cols] = imputer_int.fit_transform(df[int_cols])

    # Fill float32 columns
    if len(float_cols) > 0:
        imputer_float = SimpleImputer(strategy="median")
        df[float_cols] = imputer_float.fit_transform(df[float_cols])

    return df

def hashing_trick(df, columns, n_features=32):
    # Apply hashing trick encoding
    hasher = FeatureHasher(n_features=n_features, input_type='string')
    for col in columns:
        if col in df.columns:
            hashed = hasher.transform(df[col].astype(str).values.reshape(-1, 1))
            hashed_df = pd.DataFrame(hashed.toarray(), columns=[f"{col}_hash_{i}" for i in range(hashed.shape[1])])
            df = df.drop(col, axis=1)
            df = pd.concat([df.reset_index(drop=True), hashed_df.reset_index(drop=True)], axis=1)
    return df

def one_hot_encode_columns(df, columns):
    """
    Convert several categorical columns using One-Hot Encoding
    """
    for col in columns:
        if col in df.columns:
            # Create the encoder
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

            # Fit and transform
            encoded = encoder.fit_transform(df[[col]])

            # Create a DataFrame with clear column names
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out([col])
            )

            # Join the data and remove the original column
            df = pd.concat(
                [df.drop(columns=[col]).reset_index(drop=True),
                 encoded_df.reset_index(drop=True)],
                axis=1
            )
    return df


def replace_errorIPvalue_with_zero(df, column_name="src_bytes"):
    # Replace "0.0.0.0" strings with 0
    df[column_name] = df[column_name].replace("0.0.0.0", 0)
    
    # Convert the column to integer
    df[column_name] = df[column_name].astype(int)
    
    return df

def standardize_df(df):
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    return df


input_csv = 'your_own_filepath\Ton_IoT_Backdoor_dataset.csv'
output_csv = 'your_own_filepath\Ton_IoT_Backdoor_dataset_cleaned_feature.csv'

df = pd.read_csv(input_csv)

# Fix the Zeek display error in the bytes column
df = replace_errorIPvalue_with_zero(df)

# Replace missing values with NaN
df.replace(r'^\s*$', np.nan, regex=True)

# Replace '-' with NaN
df.replace('-', np.nan, inplace=True)

# Remove columns A, B, and C if they exist
columns_to_drop = [
    'ts',
    'src_ip',
    'dst_ip','type',
    'http_trans_depth',
    'http_method',
    'http_uri',
    'http_referrer',
    'http_version',
    'http_request_body_len',
    'http_response_body_len',
    'http_status_code',
    'http_user_agent',
    'http_orig_mime_types',
    'http_resp_mime_types',
    'dns_query',
    'dns_qclass',
    'dns_qtype',
    'dns_rcode',
    'dns_AA',
    'dns_RD',
    'dns_RA',
    'dns_rejected',
    'ssl_subject', 
    'ssl_issuer'
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Encode proto and service
df = encode_proto_features(df)

# Encode categorical columns with one-hot encoding
categorical_columns = ['weird_name', 'weird_notice', 'weird_addl', 
                       'ssl_resumed', 'ssl_established']
df = one_hot_encode_columns(df, categorical_columns)

# Columns to encode with hashing trick
hashing_columns = [
        'service', 'conn_state', 'ssl_version', 'ssl_cipher'
]

df = hashing_trick(df, hashing_columns)

# Move the label column to the end
# Check if the 'label' column exists
if 'label' in df.columns:
    # Create a list with all columns except 'label'
    cols = [c for c in df.columns if c != 'label']
    # Add 'label' to the end
    cols.append('label')
    # Reorder the DataFrame
    df = df[cols]

# Delete first line of the dataset (header: column title)
df = df.iloc[1:].reset_index(drop=True)
df = standardize_df(df)

df.to_csv(output_csv, index=False)

print("End of preprocessing")
