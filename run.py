import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#read the csv files train_data.csv and testing_data.csv into pandaas dataframes
#df_train and df_test (df -> data frame)
df_train = pd.read_csv('data/train_data.csv')
df_test = pd.read_csv('data/testing_data.csv')

# Combine train and test data to ensure consistent encoding
combined_df = pd.concat([df_train, df_test])

# Preprocess the combined data
#transform the Location and AQI_Class from categorical to numeric values
label_encoder = LabelEncoder()
combined_df['Location'] = label_encoder.fit_transform(combined_df['Location'])
combined_df['AQI_Class'] = label_encoder.fit_transform(combined_df['AQI_Class'])

# Perform one-hot encoding for 'Location' (or 'City' in your data)
city_encoder = OneHotEncoder()
city_encoded = city_encoder.fit_transform(combined_df[['Location']])
city_encoded_df = pd.DataFrame(city_encoded.toarray(), columns=[f"Location_{int(i)}" for i in range(city_encoded.shape[1])])

# Separate train and test data after encoding
# index the original df_train and df_test dataFrames for later use
train_idx = df_train.index
test_idx = df_test.index

# separate the processed data beck into df_train_processed and df_test_processed
df_train_processed = city_encoded_df.loc[train_idx]
df_test_processed = city_encoded_df.loc[test_idx]

# Separate features and target for training data
X_train = df_train_processed
Y_train = df_train['AQI'].values

X_train = np.mat(X_train)
Y_train = np.mat(Y_train).reshape((Y_train.shape[0], 1))

# Obtain the weight matrix
def getWeight(query_point, X_train, tau):
    M = X_train.shape[0]
    W = np.mat(np.eye(M))

    for i in range(M):
        xi = X_train[i]
        x = query_point
        W[i, i] = np.exp((np.sum(xi - x) ** 2) / (
                    -2 * tau * tau))  # Compute the weight for each query point and store it in the diagonal Weight matrix

    return W


def predict(X_train, Y_train, query_point, tau):
    ones = np.ones((X_train.shape[0], 1))
    X_ = np.hstack((X_train, ones))

    qx = np.hstack((query_point, np.ones((1, 1))))

    W = getWeight(qx, X_, tau)

    theta = np.linalg.pinv(X_.T * (W * X_)) * (X_.T * (W * Y_train))
    pred = np.dot(qx, theta)
    return theta, pred


# Preprocess the testing data
df_test_processed = city_encoded_df.loc[test_idx]

# Perform predictions
with open("result.csv", 'w') as f:
    f.write("Id,target\n")
    for i in range(df_test_processed.shape[0]):
        theta, pred = predict(X_train, Y_train, np.mat(df_test_processed.iloc[i, :].values), 0.1)
        f.write(f"{i+1},{pred[0, 0]}\n")

print("Predictions saved to result.csv")




