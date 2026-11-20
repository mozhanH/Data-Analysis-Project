"""Audit helpers used by Phase 2. These do not change or silently discard rows."""
from __future__ import annotations
import hashlib
import re
import numpy as np
import pandas as pd
from .cleaning import has_value


def _norm_text(value) -> str:
    return re.sub(r'\s+', ' ', str(value if value is not None else '').casefold()).strip()


def audit_duplicate_signatures(df: pd.DataFrame) -> pd.DataFrame:
    """Add Phase-1-equivalent fingerprint and conservative no-timestamp signature.

    The no-timestamp signature is a *diagnostic*, not proof of duplicate submissions.
    Rows are not dropped by this function.
    """
    out = df.copy()
    raw = out.get('combined_text', pd.Series('', index=out.index)).fillna('').map(_norm_text)
    columns = ['dataset_category', '_join_key', 'user_id', 'rating']
    fields = [out.get(c, pd.Series('', index=out.index)).fillna('').astype(str) for c in columns]
    timestamps = out.get('timestamp', pd.Series('', index=out.index)).fillna('').astype(str)

    def digest(values):
        return hashlib.sha256('\x1f'.join(values).encode('utf-8', 'replace')).hexdigest()

    out['review_fingerprint'] = [digest((*values, stamp, text)) for *values, stamp, text in zip(*fields, timestamps, raw)]
    out['review_signature_no_time'] = [digest((*values, text)) for *values, text in zip(*fields, raw)]
    out['fingerprint_duplicate_flag'] = out['review_fingerprint'].duplicated(keep=False)
    out['possible_resubmission_flag'] = out['review_signature_no_time'].duplicated(keep=False)
    return out


def duplicate_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    entries = []
    for col, label in [('review_fingerprint','Exact normalized review fingerprint'),
                       ('review_signature_no_time','Same reviewer, product, rating, text (timestamps may differ)')]:
        duplicated = df[col].duplicated(keep=False)
        counts = df.loc[duplicated, col].value_counts()
        entries.append({'signature':label, 'duplicate_rows':int(duplicated.sum()),
                        'duplicate_groups':int(len(counts)),
                        'largest_group':int(counts.max()) if len(counts) else 0})
    return pd.DataFrame(entries)


def cross_split_duplicate_audit(df: pd.DataFrame, id_col='review_signature_no_time', split_col='split'):
    """Show groups represented in >1 split. They should be examined before evaluation."""
    grouped = df.groupby(id_col, dropna=False)[split_col].nunique()
    multi = grouped[grouped > 1].index
    rows = df[df[id_col].isin(multi)]
    summary = {
        'signature': id_col,
        'cross_split_groups': int(len(multi)),
        'cross_split_rows': int(len(rows)),
    }
    return pd.DataFrame([summary]), rows


def missing_value_audit(df: pd.DataFrame):
    """Report nulls separately from semantically empty fields."""
    columns = ['helpful_vote','rating','timestamp','title','text','verified_purchase',
               'product_price','product_description','product_categories','images','product_images']
    rows=[]
    for col in columns:
        if col not in df:
            continue
        s=df[col]
        nulls=s.isna()
        usable=s.map(has_value)
        rows.append({'field':col, 'null_rows':int(nulls.sum()),
                     'null_percent':float(nulls.mean()*100),
                     'empty_or_unusable_rows':int((~usable).sum()),
                     'empty_or_unusable_percent':float((~usable).mean()*100),
                     'non_null_but_empty_rows':int((~usable & ~nulls).sum())})
    if 'product_price_num' in df:
        s=df['product_price_num']
        rows.append({'field':'product_price_num (parsed)',
                     'null_rows':int(s.isna().sum()),'null_percent':float(s.isna().mean()*100),
                     'empty_or_unusable_rows':int(s.isna().sum()),
                     'empty_or_unusable_percent':float(s.isna().mean()*100),
                     'non_null_but_empty_rows':0})
    return pd.DataFrame(rows)


def review_quality_audit(df: pd.DataFrame):
    rating=pd.to_numeric(df.get('rating_num',pd.Series(index=df.index,dtype=float)),errors='coerce')
    votes=pd.to_numeric(df.get('helpful_vote_num',pd.Series(index=df.index,dtype=float)),errors='coerce')
    timestamp=pd.to_datetime(df.get('timestamp_dt',pd.Series(index=df.index,dtype=object)),utc=True,errors='coerce')
    return pd.DataFrame([
        {'check':'Invalid nonmissing rating outside 1–5','affected_rows':int((rating.notna() & ~rating.between(1,5)).sum())},
        {'check':'Missing or unparseable rating','affected_rows':int(rating.isna().sum())},
        {'check':'Negative helpful-vote count','affected_rows':int((votes < 0).sum())},
        {'check':'Noninteger nonmissing helpful-vote count','affected_rows':int((votes.notna() & (votes % 1 != 0)).sum())},
        {'check':'Missing helpful-vote target','affected_rows':int(votes.isna().sum())},
        {'check':'Missing or unparseable timestamp','affected_rows':int(timestamp.isna().sum())},
        {'check':'Missing or empty combined cleaned review text','affected_rows':int(df['cleaned_text'].fillna('').eq('').sum())},
    ])


def distinctive_terms_by_cluster(tfidf, labels, vectorizer, top_n=12):
    """Mean TF-IDF difference against all other clusters, not just raw frequency."""
    labels=np.asarray(labels)
    terms=np.asarray(vectorizer.get_feature_names_out())
    total=np.asarray(tfidf.mean(axis=0)).ravel()
    rows=[]
    n=len(labels)
    for c in sorted(np.unique(labels)):
        idx=np.flatnonzero(labels==c)
        mean_in=np.asarray(tfidf[idx].mean(axis=0)).ravel()
        mean_out=(total*n-mean_in*len(idx))/max(1,n-len(idx))
        diff=mean_in-mean_out
        top=np.argsort(diff)[::-1][:top_n]
        for rank,j in enumerate(top,1):
            rows.append({'cluster':int(c),'rank':rank,'term':terms[j],
                         'mean_tfidf_in':float(mean_in[j]),
                         'mean_tfidf_out':float(mean_out[j]),
                         'difference':float(diff[j])})
    return pd.DataFrame(rows)


def cluster_size_and_drift(df_list):
    """df_list=(split_name, frame, labels) tuples; displays share by split."""
    parts=[]
    all_labels=sorted(set(int(c) for _,_,labels in df_list for c in np.unique(labels)))
    for name,frame,labels in df_list:
        count=pd.Series(np.asarray(labels)).value_counts().reindex(all_labels,fill_value=0).sort_index()
        for cluster,n in count.items():
            parts.append({'partition':name,'cluster':int(cluster),'count':int(n),
                          'share':float(n/len(frame))})
    return pd.DataFrame(parts)
