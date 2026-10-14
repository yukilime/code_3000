import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    quasi_ident = ["age", "gender", "zip3"]

    aux_counts = aux_df.groupby(quasi_ident).size()
    unique_keys = aux_counts[aux_counts == 1].index 

    anon_with_keys = anon_df.set_index(quasi_ident) 
    aux_with_keys = aux_df.set_index(quasi_ident) 

    matches = anon_with_keys.join( aux_with_keys[["name"]], how="inner", rsuffix="_aux" ).reset_index() 
    matches = matches.set_index(quasi_ident) 
    matches = matches.loc[matches.index.isin(unique_keys)] 
    matches = matches.reset_index()
    return matches[["anon_id", "name"]].rename( columns={"name": "matched_name"} )


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return matches_df["anon_id"].nunique() / len(anon_df)
