"""Train-only text + structured preprocessing for the Amazon Reviews project."""
from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, Normalizer, OneHotEncoder, StandardScaler
from .features import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def select_text_column(X):
    return X['cleaned_text'].fillna('').astype(str)


def select_structured_columns(X, numeric_features=None, categorical_features=None):
    n = NUMERIC_FEATURES if numeric_features is None else numeric_features
    c = CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    return X[list(n) + list(c)].copy()


def _encoder():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=True)


def build_structured_preprocessor(numeric_features=None, categorical_features=None):
    n = list(NUMERIC_FEATURES if numeric_features is None else numeric_features)
    c = list(CATEGORICAL_FEATURES if categorical_features is None else categorical_features)
    transformers = []
    if n:
        transformers.append(('numeric',Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler(with_mean=False)),
        ]), n))
    if c:
        transformers.append(('categorical',Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
            ('onehot',_encoder()),
        ]),c))
    if not transformers:
        raise ValueError('At least one structured feature is required')
    return Pipeline([
        ('columns',ColumnTransformer(transformers=transformers,remainder='drop',sparse_threshold=1.0)),
        ('row_normalize',Normalizer(copy=False)),
    ])


def build_phase2_preprocessing_pipeline(
    max_tfidf_features=15000, min_df=3, max_df=0.98, ngram_range=(1,2),
    numeric_features=None, categorical_features=None,
    text_weight=1.0, structured_weight=1.0,
):
    n=list(NUMERIC_FEATURES if numeric_features is None else numeric_features)
    c=list(CATEGORICAL_FEATURES if categorical_features is None else categorical_features)
    text_pipeline=Pipeline([
        ('select_text',FunctionTransformer(select_text_column,validate=False)),
        ('tfidf',TfidfVectorizer(lowercase=True,strip_accents='unicode',
                                  ngram_range=ngram_range,min_df=min_df,max_df=max_df,
                                  max_features=max_tfidf_features,sublinear_tf=True,
                                  token_pattern=r'(?u)\b\w\w+\b')),
        ('normalize',Normalizer(copy=False)),
    ])
    structure=Pipeline([
        ('select_structured',FunctionTransformer(
            select_structured_columns, validate=False,
            kw_args={'numeric_features':n,'categorical_features':c})),
        ('preprocess',build_structured_preprocessor(n,c)),
    ])
    return FeatureUnion(
        transformer_list=[('text',text_pipeline),('structured',structure)],
        transformer_weights={'text':float(text_weight),'structured':float(structured_weight)},
    )
