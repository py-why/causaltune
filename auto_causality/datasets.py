import pandas as pd
import numpy as np
from auto_causality.utils import featurize


def synth_ihdp():
    # load raw data
    data = pd.read_csv(
        "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv",
        header=None,
    )
    col = [
        "treatment",
        "y_factual",
        "y_cfactual",
        "mu0",
        "mu1",
    ]
    for i in range(1, 26):
        col.append("x" + str(i))
    data.columns = col
    # drop the columns we don't care about
    ignore_patterns = ["y_cfactual", "mu"]
    ignore_cols = [c for c in data.columns if any([s in c for s in ignore_patterns])]
    data = data.drop(columns=ignore_cols)
    return data

def ibm_causallib_acic(condition):
    # load data from IBM causallib 
    # condition: an integer in [1,20], see https://github.com/IBM/causallib/blob/master/causallib/datasets/data/acic_challenge_2016/README.md

    covariates =  pd.read_csv("https://raw.githubusercontent.com/IBM/causallib/master/causallib/datasets/data/acic_challenge_2016/x.csv")
    url = f'https://raw.githubusercontent.com/IBM/causallib/master/causallib/datasets/data/acic_challenge_2016/zymu_{condition}.csv'
    z_y_mu = pd.read_csv(url)
    z_y_mu['y_factual'] = z_y_mu.apply(lambda row: row['y1'] if row['z'] else row['y0'],axis = 1)
    data = pd.concat([z_y_mu['z'], z_y_mu['y_factual'], covariates],axis = 1)
    data.rename(columns = {'z': 'treatment'}, inplace = True)

    return data

def preprocess_dataset(data: pd.DataFrame) -> tuple:
    """preprocesses dataset for causal inference

    Args:
        data (pd.DataFrame): a dataset for causal inference

    Returns:
        tuple: dataset, features_x, features_w, list of targets, name of treatment
    """

    # prepare the data

    treatment = "treatment"
    targets = ["y_factual"]  # it's good to allow multiple ones
    features = [c for c in data.columns if c not in [treatment] + targets]

    data[treatment] = data[treatment].astype(int)
    # this is a trick to bypass some DoWhy/EconML bugs
    data["random"] = np.random.randint(0, 2, size=len(data))

    used_df = featurize(
        data,
        features=features,
        exclude_cols=[treatment] + targets,
        drop_first=False,
    )
    used_features = [c for c in used_df.columns if c not in [treatment] + targets]

    # Let's treat all features as effect modifiers
    features_X = [f for f in used_features if f != "random"]
    features_W = [f for f in used_features if f not in features_X]

    return used_df, features_X, features_W, targets, treatment
