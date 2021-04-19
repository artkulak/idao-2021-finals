import configparser
import pathlib as path

import numpy as np
import pandas as pd
import joblib
import os
import scipy
import scipy.stats
from catboost import CatBoostClassifier


def get_feature_total(df, col_name):
    return data['client_id'].map(df.groupby(['client_id', col_name]).size().index.get_level_values('client_id').value_counts()).fillna(0)


def get_feature_most_common(df, col_name, fill_na_value):
    return data['client_id'].map(df.groupby('client_id')[col_name].agg(lambda x: scipy.stats.mode(x)[0][0])).fillna(fill_na_value)


def get_feature_max(df, col_name, fill_na_value):
    return data['client_id'].map(df.groupby('client_id')[col_name].max()).fillna(fill_na_value)


def get_feature_min(df, col_name, fill_na_value):
    return data['client_id'].map(df.groupby('client_id')[col_name].min()).fillna(fill_na_value)


def get_feature_mean(df, col_name, fill_na_value):
    return data['client_id'].map(df.groupby('client_id')[col_name].mean()).fillna(fill_na_value)


def get_feature_std(df, col_name, fill_na_value):
    return data['client_id'].map(df.groupby('client_id')[col_name].std()).fillna(fill_na_value)


def get_feature_max_min(df, col_name, fill_na_value):
    return data['client_id'].map(df.groupby('client_id')[col_name].agg(lambda x: x.max() - x.min())).fillna(fill_na_value)


def get_feature_timedelta(df, col_name):
    return data['client_id'].map(df.groupby('client_id')[col_name].agg(lambda x: (x.max() - x.min()).days)).fillna(-1)


def get_feature_diff(df, col_name1, col_name2, fill_na_value):
    return data['client_id'].map(df.groupby('client_id')[col_name1].sum() - df.groupby('client_id')[col_name2].sum()).fillna(fill_na_value)


def get_feature_rate(df, col_name1, col_name2, fill_na_value):
    return data['client_id'].map(df.groupby('client_id')[col_name1].sum() / (df.groupby('client_id')[col_name2].sum() + 1e-12)).fillna(fill_na_value)


def create_features_transactions(data):
    
    transactions = pd.read_csv(INPUT_PATH / 'trxn.csv')
    dict_merchant_category_code = pd.read_csv('dict_mcc.csv')
    
    transactions['mcc_cd'] = transactions['mcc_cd'].fillna(-2)
    transactions['txn_city'] = transactions['txn_city'].fillna('<UNK>')
    transactions['tsp_name'] = transactions['tsp_name'].fillna('<UNK>')
    transactions['txn_comment_2'] = transactions['txn_comment_2'].fillna('<UNK>')

    transactions = transactions.merge(dict_merchant_category_code, on='mcc_cd', how='left')
    del dict_merchant_category_code
    transactions['brs_mcc_group'] = transactions['brs_mcc_group'].fillna('<UNK>')
    transactions['brs_mcc_subgroup'] = transactions['brs_mcc_subgroup'].fillna('<UNK>')
    
    data['total_transactions'] = data['client_id'].map(transactions.groupby('client_id').size()).fillna(0)
#     data['total_transactions_cards'] = get_feature_total(transactions, 'card_id')

#     data['total_transaction_amount'] = data['client_id'].map(transactions.groupby('client_id')['tran_amt_rur'].sum()).fillna(0) # add monthly, daily, etc
    data['mean_transaction_amt'] = get_feature_mean(transactions, 'tran_amt_rur', -1) # add monthly, daily, etc
    data['std_transaction_amount'] = get_feature_std(transactions, 'tran_amt_rur', -1) # add monthly, daily, etc
    
    data['total_transactions_mcc_cd'] = get_feature_total(transactions, 'mcc_cd')
    data['total_transactions_share_mcc_cd'] = (data['total_transactions_mcc_cd'] / data['total_transactions']).fillna(0)
#     data['most_common_transactions_mcc_cd'] = get_feature_most_common(transactions, 'mcc_cd', -1)
    
    data['total_transactions_merchant_cd'] = get_feature_total(transactions, 'merchant_cd')
    data['total_share_transactions_merchant_cd'] = (data['total_transactions_merchant_cd'] / data['total_transactions']).fillna(0)
#     data['most_common_transactions_merchant_cd'] = get_feature_most_common(transactions, 'merchant_cd', -1)
    
    data['total_transactions_txn_city'] = get_feature_total(transactions, 'txn_city')
    data['total_share_transactions_txn_city'] = (data['total_transactions_txn_city'] / data['total_transactions']).fillna(0)
    data['most_common_transactions_txn_city'] = get_feature_most_common(transactions, 'txn_city', '<unknown>')
    
    data['total_transactions_tsp_name'] = get_feature_total(transactions, 'tsp_name')
    data['total_share_transactions_tsp_name'] = (data['total_transactions_tsp_name'] / data['total_transactions']).fillna(0)
    data['most_common_transactions_tsp_name'] = get_feature_most_common(transactions, 'tsp_name', '<unknown>')
    
#     data['total_transactions_txn_comment_1'] = get_feature_total(transactions, 'txn_comment_1')
#     data['most_common_transactions_txn_comment_1'] = get_feature_most_common(transactions, 'txn_comment_1', '<unknown>')
    
#     data['total_transactions_txn_comment_2'] = get_feature_total(transactions, 'txn_comment_2')
#     data['most_common_transactions_txn_comment_2'] = get_feature_most_common(transactions, 'txn_comment_2', '<unknown>')
    
#     data['total_transactions_brs_mcc_group'] = get_feature_total(transactions, 'brs_mcc_group')
#     data['most_common_transactions_brs_mcc_group'] = get_feature_most_common(transactions, 'brs_mcc_group', '<unknown>')
    
#     data['total_transactions_brs_mcc_subgroup'] = get_feature_total(transactions, 'brs_mcc_subgroup')
#     data['most_common_transactions_brs_mcc_subgroup'] = get_feature_most_common(transactions, 'brs_mcc_subgroup', '<unknown>')
    
    del transactions

    return data


def create_features_aum(data):
    
    assets_under_management = pd.read_csv(INPUT_PATH / 'aum.csv')
    
    data['total_aum'] = data['client_id'].map(assets_under_management.groupby('client_id').size()).fillna(0)
    
#     data['total_aum_product_code'] = get_feature_total(assets_under_management, 'product_code')
#     data['most_common_aum_product_code'] = get_feature_most_common(assets_under_management, 'product_code', '<unknown>').value_counts()
    
    data['mean_aum_balance_rur_amt'] = get_feature_mean(assets_under_management, 'balance_rur_amt', -1)
    data['std_aum_balance_rur_amt'] = get_feature_std(assets_under_management, 'balance_rur_amt', -1)
    data['max_min_aum_balance_rur_amt'] = get_feature_max_min(assets_under_management, 'balance_rur_amt', -1)
    
    del assets_under_management
    
    return data


def create_features_balance(data):
    
    balance = pd.read_csv(INPUT_PATH / 'balance.csv')
    
    balance['crncy_cd'] = balance['crncy_cd'].fillna(-2)
    balance['prod_cat_name'] = balance['prod_cat_name'].fillna('<UNK>')
    balance['prod_group_name'] = balance['prod_group_name'].fillna('<UNK>')
    
    data['total_balance'] = data['client_id'].map(balance.groupby('client_id').size()).fillna(0)
    
#     data['total_balance_crncy_cd'] = get_feature_total(balance, 'crncy_cd')
#     data['most_common_balance_crncy_cd'] = get_feature_most_common(balance, 'crncy_cd', -1)
    
#     data['total_balance_eop_bal_sum_rur'] = get_feature_total(balance, 'eop_bal_sum_rur')
    data['total_share_balance_eop_bal_sum_rur'] = (get_feature_total(balance, 'eop_bal_sum_rur') / data['total_balance']).fillna(0)
    data['mean_balance_eop_bal_sum_rur'] = get_feature_mean(balance, 'eop_bal_sum_rur', -9999)
    data['std_balance_eop_bal_sum_rur'] = get_feature_std(balance, 'eop_bal_sum_rur', -9999)
    
    data['total_balance_min_bal_sum_rur'] = get_feature_total(balance, 'min_bal_sum_rur')
    data['total_share_balance_min_bal_sum_rur'] = (data['total_balance_min_bal_sum_rur'] / data['total_balance']).fillna(0)
    data['mean_balance_min_bal_sum_rur'] = get_feature_mean(balance, 'min_bal_sum_rur', -9999)
    data['std_balance_min_bal_sum_rur'] = get_feature_std(balance, 'min_bal_sum_rur', -9999)
    
    data['total_balance_max_bal_sum_rur'] = get_feature_total(balance, 'max_bal_sum_rur')
    data['total_share_balance_max_bal_sum_rur'] = (data['total_balance_max_bal_sum_rur'] / data['total_balance']).fillna(0)
    data['mean_balance_max_bal_sum_rur'] = get_feature_mean(balance, 'max_bal_sum_rur', -9999)
    data['std_balance_max_bal_sum_rur'] = get_feature_std(balance, 'max_bal_sum_rur', -9999)
    
    data['total_balance_avg_bal_sum_rur'] = get_feature_total(balance, 'avg_bal_sum_rur')
    data['total_share_balance_avg_bal_sum_rur'] = (data['total_balance_avg_bal_sum_rur'] / data['total_balance']).fillna(0)
    data['mean_balance_avg_bal_sum_rur'] = get_feature_mean(balance, 'avg_bal_sum_rur', -9999)
#     data['std_balance_avg_bal_sum_rur'] = get_feature_std(balance, 'avg_bal_sum_rur', -9999)
    data['max_min_balance_avg_bal_sum_rur'] = get_feature_max_min(balance, 'avg_bal_sum_rur', -9999)
    
#     data['total_balance_prod_cat_name'] = get_feature_total(balance, 'prod_cat_name')
#     data['most_common_balance_prod_cat_name'] = get_feature_most_common(balance, 'prod_cat_name', '<unknown>')
    
    data['total_balance_prod_group_name'] = get_feature_total(balance, 'prod_group_name')
#     data['most_common_balance_prod_group_name'] = get_feature_most_common(balance, 'prod_group_name', '<unknown>')
    
    del balance

    return data


def create_features_client(data):
    
    client = pd.read_csv(INPUT_PATH / 'client.csv')
    
    client = client.rename(columns={
        'gender': 'client_gender',
        'age': 'client_age',
        'region': 'client_region',
        'city': 'client_city',
        'citizenship': 'client_citizenship',
        'education': 'client_education',
        'job_type': 'client_job_type'
    })
    
    data = data.merge(client, on='client_id')
#     data['match_client_region-region_cd'] = (data['client_region'] == data['region_cd']).astype(int)
    data = data.drop(['client_citizenship', 'client_job_type', 'client_gender'], axis=1)
    
    del client
    
    return data


def create_features_campaigns(data):

    campaigns = pd.read_csv(INPUT_PATH / 'com.csv')
    
    campaigns['prod'] = campaigns['prod'].fillna('<UNK>')
    
    data['total_campaigns'] = data['client_id'].map(campaigns.groupby('client_id').size()).fillna(0)
    
#     data['total_campaigns_agr_flg'] = get_feature_total(campaigns, 'agr_flg')
    data['mean_campaigns_agr_flg'] = get_feature_mean(campaigns, 'agr_flg', -1)
    
#     data['total_campaigns_otkaz'] = get_feature_total(campaigns, 'otkaz')
    data['mean_campaigns_otkaz'] = get_feature_mean(campaigns, 'otkaz', -1)
    
#     data['total_campaigns_dumaet'] = get_feature_total(campaigns, 'dumaet')
    data['mean_campaigns_dumaet'] = get_feature_mean(campaigns, 'dumaet', -1)
    
#     data['total_campaigns_ring_up_flg'] = get_feature_total(campaigns, 'ring_up_flg')
#     data['most_common_campaigns_ring_up_flg'] = get_feature_most_common(campaigns, 'ring_up_flg', -1)
    
#     data['total_campaigns_count_comm'] = get_feature_total(campaigns, 'count_comm')
#     data['most_common_campaigns_count_comm'] = get_feature_most_common(campaigns, 'count_comm', -1)
    
#     data['total_campaigns_channel'] = get_feature_total(campaigns, 'channel')
#     data['most_common_campaigns_channel'] = get_feature_most_common(campaigns, 'channel', '<unknown>')
    
#     data['total_campaigns_prod'] = get_feature_total(campaigns, 'prod')
    data['most_common_campaigns_prod'] = get_feature_most_common(campaigns, 'prod', '<unknown>')
    
#     data['diff_campaigns_otkaz-agr_flg'] = get_feature_diff(campaigns, 'otkaz', 'agr_flg', -999)
    
    data['rate_campaigns_otkaz-count_comm'] = get_feature_rate(campaigns, 'otkaz', 'count_comm', -999)
    data['rate_campaigns_agr_flg-count_comm'] = get_feature_rate(campaigns, 'agr_flg', 'count_comm', -999)
    data['rate_campaigns_not_ring_up_flg-count_comm'] = get_feature_rate(campaigns, 'not_ring_up_flg', 'count_comm', -999)
    data['rate_campaigns_ring_up_flg-count_comm'] = get_feature_rate(campaigns, 'ring_up_flg', 'count_comm', -999)
    
    del campaigns
    
    return data


def create_features_deals(data):
    
    deals = pd.read_csv(INPUT_PATH / 'deals.csv')
    
    deals['crncy_cd'] = deals['crncy_cd'].fillna(-2)
    deals['agrmnt_rate_active'] = deals['agrmnt_rate_active'].fillna(-2)
    deals['agrmnt_rate_passive'] = deals['agrmnt_rate_passive'].fillna(-2)
    deals['agrmnt_sum_rur'] = deals['agrmnt_sum_rur'].fillna(-2)
    deals['prod_type_name'] = deals['prod_type_name'].fillna('<UNK>')
    deals['argmnt_close_start_days'] = (pd.to_datetime(deals['agrmnt_close_dt']) - pd.to_datetime(deals['agrmnt_start_dt'])).dt.days.fillna(-2)
    
    data['total_deals'] = data['client_id'].map(deals.groupby('client_id').size()).fillna(0)
    
#     data['total_deals_crncy_cd'] = get_feature_total(deals, 'crncy_cd')
#     data['most_common_deals_crncy_cd'] = get_feature_most_common(deals, 'crncy_cd', -1)
    
    data['total_deals_agrmnt_rate_active'] = get_feature_total(deals, 'agrmnt_rate_active')
    data['max_deals_agrmnt_rate_active'] = get_feature_max(deals, 'agrmnt_rate_active', -1)
    
#     data['total_deals_agrmnt_rate_passive'] = get_feature_total(deals, 'agrmnt_rate_passive')
    data['max_deals_agrmnt_rate_passive'] = get_feature_max(deals, 'agrmnt_rate_passive', -1)
    
    data['total_deals_agrmnt_sum_rur'] = get_feature_total(deals, 'agrmnt_sum_rur')
    data['mean_deals_agrmnt_sum_rur'] = get_feature_mean(deals, 'agrmnt_sum_rur', -1)
    data['std_deals_agrmnt_sum_rur'] = get_feature_std(deals, 'agrmnt_sum_rur', -1)
    
    data['total_deals_prod_type_name'] = get_feature_total(deals, 'prod_type_name')
    data['most_common_deals_prod_type_name'] = get_feature_most_common(deals, 'prod_type_name', '<unknown>')
    
    data['total_deals_argmnt_close_start_days'] = get_feature_total(deals, 'argmnt_close_start_days')
    data['max_deals_argmnt_close_start_days'] = get_feature_max(deals, 'argmnt_close_start_days', -1)
#     data['min_deals_argmnt_close_start_days'] = get_feature_min(deals, 'argmnt_close_start_days', -1)
    data['mean_deals_argmnt_close_start_days'] = get_feature_mean(deals, 'argmnt_close_start_days', -1)
    data['std_deals_argmnt_close_start_days'] = get_feature_std(deals, 'argmnt_close_start_days', -1)
    
    del deals
    
    return data


def create_features_payments(data):
    
    payments = pd.read_csv(INPUT_PATH / 'payments.csv')
    
    payments['day_dt'] = pd.to_datetime(payments['day_dt'])
    
    data['total_payments'] = data['client_id'].map(payments.groupby('client_id').size()).fillna(0)
    
    data['mean_payments_sum_rur'] = get_feature_mean(payments, 'sum_rur', -1)
    data['std_payments_sum_rur'] = get_feature_std(payments, 'sum_rur', -1)
    data['min_payments_sum_rur'] = get_feature_min(payments, 'sum_rur', -1)
    data['max_payments_sum_rur'] = get_feature_max(payments, 'sum_rur', -1)
    
#     data['total_payments_pmnts_name'] = get_feature_total(payments, 'pmnts_name')
#     data['most_common_payments_pmnts_name'] = get_feature_most_common(payments, 'pmnts_name', '<unknown>')
    
    # payments 
#     data['last_known_salary'] = data['client_id'].map(payments.groupby('client_id').apply(lambda x: x['sum_rur'].iloc[0])).fillna(-1)
#     data['total_recieved_salary'] = data['client_id'].map(payments.groupby('client_id').apply(lambda x: x['sum_rur'].sum())).fillna(-1)
    
    data['timedelta_payments_day_dt'] = get_feature_timedelta(payments, 'day_dt')
    
    del payments
    
    return data


def create_features_appl(data):
    
    appl = pd.read_csv(INPUT_PATH / 'appl.csv')
    
    appl['appl_stts_name_dc'] = appl['appl_stts_name_dc'].fillna('<UNK>')
    appl['appl_sale_channel_name'] = appl['appl_sale_channel_name'].fillna('<UNK>')
    appl['month_end_dt'] = pd.to_datetime(appl['month_end_dt'])
    
    data['total_appl'] = data['client_id'].map(appl.groupby('client_id').size()).fillna(0)
    
#     data['total_appl_prod_group_name'] = get_feature_total(appl, 'appl_prod_group_name')
#     data['most_common_appl_prod_group_name'] = get_feature_most_common(appl, 'appl_prod_group_name', '<unknown>')
    
    data['total_appl_prod_type_name'] = get_feature_total(appl, 'appl_prod_type_name')
    data['most_common_appl_prod_type_name'] = get_feature_most_common(appl, 'appl_prod_type_name', '<unknown>')
    
#     data['total_appl_stts_name_dc'] = get_feature_total(appl, 'appl_stts_name_dc')
#     data['most_common_appl_stts_name_dc'] = get_feature_most_common(appl, 'appl_stts_name_dc', '<unknown>')
    
#     data['total_appl_sale_channel_name'] = get_feature_total(appl, 'appl_sale_channel_name')
#     data['most_common_appl_sale_channel_name'] = get_feature_most_common(appl, 'appl_sale_channel_name', '<unknown>')
    
    data['timedelta_appl_month_end_dt'] = get_feature_timedelta(appl, 'month_end_dt')
    
    del appl
    
    return data


def create_features_funnel(data):
    
    return data

def get_data(INPUT_PATH, PREPROCESSORS_PATH):
    funnel = pd.read_csv(INPUT_PATH / 'funnel.csv')
    global data 
    data = funnel.copy()

    del funnel

    data = create_features_transactions(data)
    data = create_features_aum(data)
    data = create_features_balance(data)
    data = create_features_client(data)
    data = create_features_campaigns(data)
    data = create_features_deals(data)
    data = create_features_payments(data)
    data = create_features_appl(data)
    data = create_features_funnel(data)


    return data



def main(cfg):
    # parse config
    global INPUT_PATH
    INPUT_PATH = path.Path(cfg["DATA"]["DatasetPath"])
    PREPROCESSORS_PATH = path.Path(cfg["MODEL"]["Preprocessors"])
    USER_ID = cfg["COLUMNS"]["USER_ID"]
    PREDICTION = cfg["COLUMNS"]["PREDICTION"]
    SUBMISSION_FILE = path.Path(cfg["SUBMISSION"]["FilePath"])
    # do something with data
    X = get_data(INPUT_PATH, PREPROCESSORS_PATH)
    submission = X[[USER_ID]].copy()
    X = X.drop(columns = [USER_ID])

    # catboost
    cat_feats = []
    for c in X.columns:
        col_type = X[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            X[c] = X[c].astype('str')
            cat_feats.append(np.argwhere(X.columns == c)[0][0])

    CATBOOST_MODEL_DIR = path.Path('catboost_models')
    LIGHTGBM_MODEL_DIR = path.Path('lightgbm_models')
    preds = np.zeros((X.shape[0], len(os.listdir(CATBOOST_MODEL_DIR) + os.listdir(LIGHTGBM_MODEL_DIR))))
    for i, model_name in enumerate(os.listdir(CATBOOST_MODEL_DIR)):
        model = CatBoostClassifier(cat_features = cat_feats)
        model.load_model(str(CATBOOST_MODEL_DIR / model_name))
        preds[:, i] = model.predict_proba(X)[:, 1]

    # lightgbm
    for c in X.columns:
        col_type = X[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            X[c] = X[c].astype('category')
    ooffset = len(os.listdir(CATBOOST_MODEL_DIR))
    for i, model_name in enumerate(os.listdir(LIGHTGBM_MODEL_DIR)):
        model = joblib.load(LIGHTGBM_MODEL_DIR / model_name)
        preds[:,ooffset+i] = model.predict_proba(X)[:, 1]

    
    # submission[PREDICTION] = np.random.choice([0, 1], len(submission))
    submission[PREDICTION] = (np.mean(preds, axis = 1) > 0.16).astype(int)
    submission.to_csv(SUBMISSION_FILE, index=False)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")
    main(cfg=config)
