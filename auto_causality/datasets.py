import pandas as pd
from .data_utils import preprocess_mixed


def nhefs() -> pd.DataFrame:
    """loads the NHEFS dataset
    The dataset describes the impact of quitting smoke on weight gain over a period of 11 years
    The data consists of the treatment (quit smoking yes no), the outcome (change in weight) and
    a series of covariates of which we include a subset of 9 (see below).

    If used for academic purposes, pelase consider citing the authors:
    Hernán MA, Robins JM (2020). Causal Inference: What If. Boca Raton: Chapman & Hall/CRC.

    Returns:
        pd.DataFrame: dataset with cols "treatment", "y_factual" and covariates "x1" to "x9"
    """

    df = pd.read_csv(
        "https://cdn1.sph.harvard.edu/wp-content/uploads/sites/1268/1268/20/nhefs.csv"
    )
    covariates = [
        "active",
        "age",
        "education",
        "exercise",
        "race",
        "sex",
        "smokeintensity",
        "smokeyrs",
        "wt71",
    ]

    has_missing = ["wt82"]
    missing = df[has_missing].isnull().any(axis="columns")
    df = df.loc[~missing]

    df = df[covariates + ["qsmk"] + ["wt82_71"]]
    df.rename(columns={"qsmk": "treatment", "wt82_71": "y_factual"}, inplace=True)
    df.rename(
        columns={c: "x" + str(i + 1) for i, c in enumerate(covariates)}, inplace=True
    )

    return df


def lalonde_nsw() -> pd.DataFrame:
    """loads the Lalonde NSW dataset
    The dataset described the impact of a job training programme on the real earnings
    of individuals several years later.
    The data consists of the treatment indicator (training yes no), covariates (age, race,
    academic background, real earnings 1976, real earnings 1977) and the outcome (real earnings in 1978)
    See also https://rdrr.io/cran/qte/man/lalonde.html#heading-0

    If used for academic purposes, please consider citing the authors:
    Lalonde, Robert: "Evaluating the Econometric Evaluations of Training Programs," American Economic Review,
    Vol. 76, pp. 604-620

    Returns:
        pd.DataFrame: dataset with cols "treatment", "y_factual" and covariates "x1" to "x8"
    """

    df_control = pd.read_csv(
        "https://users.nber.org/~rdehejia/data/nswre74_control.txt", sep=" "
    ).dropna(axis=1)
    df_control.columns = (
        ["treatment"] + ["x" + str(x) for x in range(1, 9)] + ["y_factual"]
    )
    df_treatment = pd.read_csv(
        "https://users.nber.org/~rdehejia/data/nswre74_treated.txt", sep=" "
    ).dropna(axis=1)
    df_treatment.columns = (
        ["treatment"] + ["x" + str(x) for x in range(1, 9)] + ["y_factual"]
    )
    df = (
        pd.concat([df_control, df_treatment], axis=0, ignore_index=True)
        .sample(frac=1)
        .reset_index(drop=True)
    )
    return df


def amazon_reviews(rating="pos") -> pd.DataFrame:
    """loads amazon reviews dataset
    The dataset describes the impact of positive (or negative) reviews for products on Amazon on sales.
    The authors distinguish between items with more than three reviews (treated) and less than three
    reviews (untreated). As the rating given by reviews might impact sales, they divide the dataset
    into products with on average positive (more than 3 starts) or negative (less than three stars)
    reviews.
    The dataset consists of 305 covariates (doc2vec features of the review text), a binary treatment
    variable (more than 3 reviews vs less than three reviews) and a continuous outcome (sales).

    If used for academic purposes, please consider citing the authors:
    @inproceedings{rakesh2018linked,
        title={Linked Causal Variational Autoencoder for Inferring Paired Spillover Effects},
        author={Rakesh, Vineeth and Guo, Ruocheng and Moraffah, Raha and Agarwal, Nitin and Liu, Huan},
        booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
        pages={1679--1682},
        year={2018},
        organization={ACM}
    }
    Args:
        rating (str, optional): choose between positive ('pos') and negative ('neg') reviews. Defaults to 'pos'.

    Returns:
        pd.DataFrame: dataset with cols "treatment", "y_factual" and covariates "x1" to "x300"
    """
    try:
        assert rating in ["pos", "neg"]
    except AssertionError:
        print(
            "you need to specify which rating dataset you'd like to load. The options are 'pos' or 'neg'"
        )
        return None

    try:
        import gdown
    except ImportError:
        gdown = None

    if rating == "pos":
        url = "https://drive.google.com/file/d/167CYEnYinePTNtKpVpsg0BVkoTwOwQfK/view?usp=sharing"
    elif rating == "neg":
        url = "https://drive.google.com/file/d/1b-MPNqxCyWSJE5uyn5-VJUwC8056HM8u/view?usp=sharing"

    if gdown:
        try:
            df = pd.read_csv("amazon_" + rating + ".csv")
        except FileNotFoundError:
            gdown.download(url, "amazon_" + rating + ".csv", fuzzy=True)
            df = pd.read_csv("amazon_" + rating + ".csv")
        df.drop(df.columns[[2, 3, 4]], axis=1, inplace=True)
        df.columns = ["treatment", "y_factual"] + ["x" + str(i) for i in range(1, 301)]
        return df
    else:
        print(
            f"""The Amazon dataset is hosted on google drive. As it's quite large, the gdown package is required to download
            the package automatically. The package can be installed via 'pip install gdown'.
            Alternatively, you can download it from the following link and store it in the datasets folder:
            {url}"""
        )
        return None


def synth_ihdp() -> pd.DataFrame:
    """loads IHDP dataset
    The Infant Health and Development Program (IHDP) dataset contains data on the impact of visits by specialists
    on the cognitive development of children. The dataset consists of 25 covariates describing various features
    of these children and their mothers, a binary treatment variable (visit/no visit) and a continuous outcome.

    If used for academic purposes, consider citing the authors:
    @article{hill2011,
        title={Bayesian nonparametric modeling for causal inference.},
        author={Hill, Jennifer},
        journal={Journal of Computational and Graphical Statistics},
        volume={20},
        number={1},
        pages={217--240},
        year={2011}
    }

    Returns:
        pd.DataFrame: dataset for causal inference with cols "treatment", "y_factual" and covariates "x1" to "x25"
    """
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


def synth_acic(condition=1) -> pd.DataFrame:
    """loads data from ACIC Causal Inference Challenge 2016
    The dataset consists of 58 covariates, a binary treatment and a continuous response.
    There are 10 simulated pairs of treatment and response, which can be selected
    with the condition argument supplied to this function.

    If used for academic purposes, consider citing the authors:
    @article{dorie2019automated,
        title={Automated versus do-it-yourself methods for causal inference: Lessons learned from a
         data analysis competition},
        author={Dorie, Vincent and Hill, Jennifer and Shalit, Uri and Scott, Marc and Cervone, Dan},
        journal={Statistical Science},
        volume={34},
        number={1},
        pages={43--68},
        year={2019},
        publisher={Institute of Mathematical Statistics}
    }

    Args:
        condition (int): in [1,10], corresponds to 10 simulated treatment/response pairs. Defaults to 1.

    Returns:
        pd.DataFrame: dataset for causal inference with columns "treatment", "y_factual" and covariates "x_1" to "x_58"
    """
    try:
        assert condition in range(1, 11)
    except AssertionError:
        print("'condition' needs to be in [1,10]")
        return None

    covariates = pd.read_csv(
        "https://raw.githubusercontent.com/IBM/causallib/"
        + "master/causallib/datasets/data/acic_challenge_2016/x.csv"
    )
    cols = covariates.columns
    covariates.rename(
        columns={c: c.replace("_", "") for c in cols},
        inplace=True,
    )
    url = (
        "https://raw.githubusercontent.com/IBM/causallib/master/causallib/"
        + f"datasets/data/acic_challenge_2016/zymu_{condition}.csv"
    )
    z_y_mu = pd.read_csv(url)
    z_y_mu["y_factual"] = z_y_mu.apply(
        lambda row: row["y1"] if row["z"] else row["y0"], axis=1
    )
    data = pd.concat([z_y_mu["z"], z_y_mu["y_factual"], covariates], axis=1)
    data.rename(columns={"z": "treatment"}, inplace=True)

    return data


def bdsianesi() -> pd.DataFrame:
    """ Loads bdsianesi (Blundell, Dearden and Sianesi - 2005) dataset
        This data extract explores the impact of education (treatment) on wages 
        for individuals in the UK. The dataset comprises 18 covariates, a multi-valued 
        treatment (ed) of: No degree, O-level, A-level & Higher degree and 
        a continuous response (wage)

        If used for academic purposes, consider citing the authors:
        @article{https://doi.org/10.1111/j.1467-985X.2004.00360.x,
            author = {Blundell, Richard and Dearden, Lorraine and Sianesi, Barbara},
            title = {Evaluating the effect of education on earnings: models, methods and results from the National Child Development Survey},
            journal = {Journal of the Royal Statistical Society: Series A (Statistics in Society)},
            volume = {168},
            number = {3},
            pages = {473-512},
            year = {2005}
        }
    """

    url = ("http://www.stata-press.com/data/r13/bdsianesi5.dta")
    
    treatment = "ed"
    outcome = "wage"
    control_val = "none"
    drop_cols = ["lwage"]

    data = pd.read_stata(url)
    df = preprocess_mixed(data, treatment, control_val, drop_cols)

    return df
