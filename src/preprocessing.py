import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from copy import deepcopy
from utils import deserialize_data, serialize_data


def drop_duplicate_data(X, y):
    """
    This function drops duplicated data from row X and y.

    Parameters
    -----------
    X : dataframe
        features of dataset

    y : series
        target of dataset

    Returns
    -------
    X : dataframe
        dropped duplicated data features of dataset

    y : dataframe
        dropped duplicated data target of dataset
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Fungsi median_imputation: parameter X haruslah bertipe DataFrame!")

    if not isinstance(y, pd.Series):
        raise TypeError("Fungsi median_imputation: parameter y haruslah bertipe DataFrame!")

    print(f"Fungis drop_duplicate_data telah divalidasi.")

    X = X.copy()
    y = y.copy()
    print(f"Fungsi drop_duplicate_data: shape dataset sebelum dropping duplicate adalah {X.shape}.")

    X_duplicate = X_train[X_train.duplicated()]
    print(f"Fungsi drop_duplicate_data: shape dari data yang duplicate adalah {X_duplicate.shape}.")

    X_clean = (X.shape[0] - X_duplicate.shape[0], X.shape[1])
    print(f"Fungsi drop_duplicate_data: shape dataset setelah drop duplicate seharusnya adalah {X_clean}.")

    X.drop_duplicates(inplace=True)
    y = y[X.index]

    print(f"Fungsi drop_duplicate_data: shape dataset setelah dropping duplicate adalah {X.shape}.")

    return X, y


def median_imputation(data, subset_data, fit):
    """
    Parameters
    -----------
    data : dataframe
        dataset to be imputed

    subset_data : list of string
        columns name

    fit : boolean
        if fit=true, this function will return median of subset_data
        if fit=false, this function will impute the data based on subset_data

    Returns
    -------
    X : dataframe
        dropped duplicated data features of dataset

    y : dataframe
        dropped duplicated data target of dataset
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Fungsi median_imputation: parameter data haruslah bertipe DataFrame!")

    if fit is True and not isinstance(subset_data, list):
        raise TypeError(
            "Fungsi median_imputation: untuk nilai parameter fit = True, subset_data harus bertipe list dan berisi "
            "daftar nama kolom yang ingin dicari nilai mediannya guna menjadi data imputasi pada kolom tersebut")

    if fit is False and not isinstance(subset_data, dict):
        raise TypeError(
            "Fungsi median_imputation: untuk nilai parameter fit = False, subset_data harus bertipe dict dan berisi "
            "key yang merupakan nama kolom beserta value yang merupakan nilai median dari kolom tersebut")

    if not isinstance(fit, bool):
        raise TypeError("Fungsi median_imputation: parameter fit haruslah bertipe boolean, bernilai True atau False.")

    print("Fungsi median_imputation: parameter telah divalidasi.")

    data = data.copy()
    subset_data = deepcopy(subset_data)

    """
    Handles fitting data
    """
    if fit is True:
        imputation_data = {}
        for subset in subset_data:
            imputation_data[subset] = data[subset].median(numeric_only=True)

        print(f"Fungsi median_imputation: proses fitting telah selesai, berikut hasilnya {imputation_data}")

        return imputation_data

    """
    Handles transforming data
    """
    print("Fungsi median_imputation: informasi count na sebelum dilakukan imputasi")
    print(data.isna().sum())
    print()

    for subset in subset_data:
        data[subset] = data[subset].fillna(subset_data[subset])

    print("Fungsi median_imputation: informasi count na setelah dilakukan imputasi.")
    print(data.isna().sum())
    print()

    return data


def create_onehot_encoder(categories, path):
    """
    create_onehot_encoder create encoder based on categories and save it in path

    :param categories:
    :param path:
    :return:
    """

    if not isinstance(categories, list):
        raise TypeError(
            "Fungsi create_onehot_encoder: parameter categories haruslah bertipe list, berisi kategori yang akan dibuat encodernya.")

    if not isinstance(path, str):
        raise TypeError(
            "Fungsi create_onehot_encoder: parameter path haruslah bertipe string, berisi lokasi pada disk komputer dimana encoder akan disimpan.")

    ohe = OneHotEncoder()
    categories_ = np.array(categories).reshape(-1, 1)
    ohe.fit(categories_)

    serialize_data(ohe, path)

    print(f"Kategori yang telah dipelajari adalah {categories_[0].tolist()}")

    return ohe


def ohe_transform(dataset, subset, prefix, ohe):
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("Fungsi ohe_transform: parameter dataset harus bertipe DataFrame!")

    if not isinstance(subset, str):
        raise TypeError("Fungsi ohe_transform: parameter ohe harus bertipe OneHotEncoder!")

    if not isinstance(prefix, str):
        raise TypeError("Fungsi ohe_transform: parameter prefix harus bertipe str")

    if not isinstance(ohe, OneHotEncoder):
        raise TypeError("Fungsi ohe_transform: parameter subset harus bertipe str!")

    try:
        dataset[subset]
    except:
        raise RuntimeError(
            "Fungsi ohe_transform: parameter subset string namun data tidak ditemukan dalam daftar kolom yang terdapat pada parameter datase")

    print("Fungsi ohe_transform: parameter telah divalidasi.")

    print(f"Fungsi ohe_transform: daftar nama kolom sebelum dilakukan pengkodean adalah {list(dataset.columns)}")

    dataset = dataset.copy()

    col_names = []
    for col in ohe.categories_[0]:
        col_name = f"{prefix}_{col}"
        col_names.append(col_name)

    encoded = pd.DataFrame(
        ohe.transform(dataset[subset].to_frame()).toarray(),
        columns=col_names,
        index=dataset[subset].index
    )

    dataset = pd.concat([dataset, encoded], axis=1)
    dataset = dataset.drop(subset, axis=1)

    print(f"Fungsi ohe_transform: daftar nama kolom setelah dilakukan pengkodean adalah {list(dataset.columns)}")

    return dataset


def fit_scaler(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler


def transform_scaler(scaler, data):
    scaled_data_raw = scaler.transform(data)
    scaled_data_frame = pd.DataFrame(scaled_data_raw, columns=data.columns, index=data.index)
    return scaled_data_frame


X_train = deserialize_data("../data/interim/X_train.pkl")
y_train = deserialize_data("../data/interim/y_train.pkl")


#### DROP DUPLICATED DATA ####
X_train_dropped, y_train_dropped = drop_duplicate_data(X_train, y_train)


#### DATA IMPUTATION ####
subset_data = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"
]
subset_data = median_imputation(X_train, subset_data, fit=True)
X_train_imputed = median_imputation(X_train, subset_data, fit=False)


#### OHE TRANSFORM ####
person_home_ownership = ["RENT", "OWN", "MORTGAGE", "OTHER"]
loan_intent = ["PERSONAL", "DEBTCONSOLIDATION", "MEDICAL", "HOMEIMPROVEMENT", "VENTURE", "EDUCATION"]
loan_grade = ["C", "E", "A", "B", "D", "F", "G"]
cb_person_default_on_file = ["Y", "N"]

ohe_person_home_ownership = create_onehot_encoder(person_home_ownership, "../models/person_home_ownership.pkl")
ohe_loan_intent = create_onehot_encoder(loan_intent, "../models/loan_intent.pkl")
ohe_loan_grade = create_onehot_encoder(loan_grade, "../models/loan_grade.pkl")
ohe_cb_person_default_on_file = create_onehot_encoder(cb_person_default_on_file, "../models/cb_person_default_on_file.pkl")

X_train = ohe_transform(X_train, "person_home_ownership", "home_ownership", ohe_person_home_ownership)
X_train = ohe_transform(X_train, "loan_intent", "loan_intent", ohe_loan_intent)
X_train = ohe_transform(X_train, "loan_grade", "grade", ohe_loan_grade)
X_train = ohe_transform(X_train, "cb_person_default_on_file", "default_on_file", ohe_cb_person_default_on_file)

scaler = fit_scaler(X_train)


def preprocess_data(x_data):
    x_data = ohe_transform(x_data, "person_home_ownership", "home_ownership", ohe_person_home_ownership)
    x_data = ohe_transform(x_data, "loan_intent", "loan_intent", ohe_loan_intent)
    x_data = ohe_transform(x_data, "loan_grade", "grade", ohe_loan_grade)
    x_data = ohe_transform(x_data, "cb_person_default_on_file", "default_on_file", ohe_cb_person_default_on_file)

    x_data_clean = transform_scaler(scaler, x_data)

    return x_data_clean
