"""
Functions:
▪ outlier_thresholds
▪ replace_with_thresholds
▪ check_outlier
▪ grab_outliers
▪ remove_outlier
▪ missing_values_table
▪ missing_vs_target
▪ label_encoder
▪ one_hot_encoder
▪ rare_analyser
▪ rare_encoder
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_outliers(dataframe, col_name, index=False, q1=0.05, q3=0.95):
    low, up = outlier_thresholds(dataframe, col_name, q1, q3)
    dataframe_with_outliers = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))]
    print(f"column: {col_name}, outlier counts: {len(dataframe_with_outliers)}, low limit: {low}, up limit: {up}")
    if len(dataframe_with_outliers) > 10:
        print(dataframe_with_outliers.head(), end="\n\n")
    else:
        print(dataframe_with_outliers, end="\n\n")

    if index:
        outlier_index = dataframe_with_outliers.index
        return outlier_index


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc, cat_cols):
    # 1'den fazla rare varsa düzeltme yap. durumu göz önünde bulunduruldu.
    # rare sınıf sorgusu 0.01'e göre yapıldıktan sonra gelen true'ların sum'ı alınıyor.
    # eğer 1'den büyük ise rare cols listesine alınıyor.
    dataframe = dataframe.copy()
    rare_columns = [col for col in cat_cols if (dataframe[col].value_counts() / len(dataframe) < 0.01).sum() > 1]

    for col in rare_columns:
        tmp = dataframe[col].value_counts() / len(dataframe)
        rare_labels = tmp[tmp < rare_perc].index
        dataframe.loc[dataframe[col].isin(rare_labels), col] = "Rare"

    return dataframe


def split_sets(dataframe, dependent_col="Outcome"):
    independent_cols = [col for col in dataframe.columns if col != dependent_col]
    return dataframe[independent_cols], dataframe[dependent_col]


def qcut_and_groupby_with_target(dataframe, target, num_col, q=10):
    dataframe = dataframe.copy()
    new_col_name = str(num_col)+"_grouped"
    dataframe[new_col_name] = pd.qcut(dataframe[num_col], q=q)
    print(dataframe.groupby(new_col_name).agg({target: ["mean", "count"]}))
    print("#########################################")


def fill_nas_with_knn(df, columns_to_be_filled, neightbors=5):
    dff = pd.get_dummies(df, drop_first=True)
    scaler = MinMaxScaler()
    dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
    imputer = KNNImputer(n_neighbors=neightbors)
    dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
    dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
    return dff[columns_to_be_filled]




