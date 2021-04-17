from scorer import calculate_metric
import pandas as pd

TMP_SUBMISSION = 'submission/test_submission.csv'
TMP_TRUTH = 'submission/test_truth.csv'

TARGET_COLUMNS = ['sale_flg', 'sale_amount']
TRUTH_COLUMNS = ['contacts', 'sale_flg', 'sale_amount']
SUBMISSION_COLS = ['client_id', 'target']


def write_submission(Y_pred, Y_test):
    tmp_sub = pd.Series(Y_pred, index=Y_test.index)
    tmp_sub = tmp_sub.reset_index()
    tmp_sub.columns = SUBMISSION_COLS
    tmp_sub.to_csv(TMP_SUBMISSION, index=False)


def write_truth(test_df):
    assert test_df.index.name == 'client_id'
    truth_test = test_df[TRUTH_COLUMNS]
    truth_test['split'] = 'public'
    truth_test.iloc[:truth_test.shape[0] // 2].split = 'private'
    truth_test.to_csv(TMP_TRUTH)

def get_score(test_df, Y_pred, Y_test):
    """
    test_df: test_df(indexed by client id)
    Y_pred: Y_pred
    Y_test: Y_test(indexed by client id)
    """
    write_truth(test_df)
    write_submission(Y_pred, Y_test)
    public, private = calculate_metric(TMP_TRUTH, TMP_SUBMISSION)
    return public, private
