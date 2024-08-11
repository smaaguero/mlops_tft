import pandas as pd
import requests
from flatten_json import flatten
from typing import List
from sklearn.pipeline import Pipeline
from .sklearn_pipes import NaNDropper, CorruptedDropper, ResetIndex, DescribeMissing
from .sklearn_pipes import TrainDropper, OutlierRemover, GetAugmentDummies


def get_challenger_data(
        api_key: str, 
        region: str
    ) -> List[str]:
    """
    Retrieves challenger-level player data for a specified region in Teamfight Tactics (TFT).

    Args:
        api_key (str): The API key for accessing Riot Games' API.
        region (str): The region code for the desired data (e.g., "na1", "euw1").

    Returns:
        List[str]: A list containing the challenger player data.
    """
    # Construct the URL for the API request
    ph_gm_url = "".join(
        [
            "https://",
            region,
            ".api.riotgames.com/tft/league/v1/challenger?queue=RANKED_TFT",
        ]
    )
    ph_gm_url = ph_gm_url + "&api_key=" + api_key

    # Make the API request and parse the JSON response
    try:
        ph_gm_resp = requests.get(ph_gm_url, timeout=127)
        gm_info = ph_gm_resp.json()
        return gm_info
    except:  # noqa: E722
        print("Request has timed out.")


def get_id(players: List[str]) -> List[str]:
    """
    Extracts summoner IDs from player data.

    Args:
        players (List[str]): A list containing player data.

    Returns:
        List[str]: A list of summoner IDs for the players.
    """
    
    player_ids = [player["summonerId"] for player in players["entries"]]
    
    return player_ids


def get_puuid(
    names: List[str], 
    region: str,
    api_key: str
) -> List[str]:
    """
    Retrieves the PUUIDs (Player Universally Unique Identifiers) for the given summoner names.

    Args:
        names (List[str]): A list of summoner names.
        region (str): The region code for the desired data.
        api_key (str): The API key for accessing Riot Games' API.

    Returns:
        List[str]: A list of PUUIDs for the specified summoner names.
    """

    puuids = []
    # Loop through each summoner name to retrieve their PUUID
    for name in names:
        puuid_url = "".join(
            [
                "https://",
                region,
                ".api.riotgames.com/tft/summoner/v1/summoners/",
            ]
        )
        puuid_url = ''.join(
            [
                puuid_url, 
                name, 
                "?api_key=",
                api_key
            ]
        )
        puuid_resp = requests.get(puuid_url, timeout=127)
        puuids += [dict(puuid_resp.json())]

    # Extract the PUUIDs from the response data
    final_puuids = []
    for i in range(len(puuids)):
        final_puuids += [puuids[i].get("puuid")]

    return final_puuids


def get_match_ids(
    puuids: List[str], 
    region_extended: str, 
    n_matches: int,
    api_key: str
) -> List[str]:
    """
    Retrieves match IDs for the specified PUUIDs.

    Args:
        puuids (List[str]): A list of PUUIDs.
        region_extended (str): The extended region code for match retrieval (e.g., "americas", "europe").
        n_matches (int): The number of matches to retrieve for each PUUID.
        api_key (str): The API key for accessing Riot Games' API.

    Returns:
        List[str]: A list of match IDs for the specified PUUIDs.
    """
    
    # Loop through each PUUID to retrieve their match IDs
    match_ids = []
    for puuid in puuids:
        match_url = "".join(
            [
                "https://",
                region_extended,
                ".api.riotgames.com/tft/match/v1/matches/by-puuid/",
            ]
        )
        match_url += "".join(
            [puuid, "/ids?start=0&count=", str(n_matches), "&api_key=", api_key]
        )
        match_resp = requests.get(match_url, timeout=127)
        match_ids += match_resp.json()

    return match_ids


def get_match_data(
    match_ids: List[str], 
    region_extended: str,
    api_key: str
) -> pd.DataFrame:
    """
    Retrieves match data for the specified match IDs.

    Args:
        match_ids (List[str]): A list of match IDs.
        region_extended (str): The extended region code for match retrieval.
        api_key (str): The API key for accessing Riot Games' API.

    Returns:
        pd.DataFrame: A DataFrame containing match data, with one row per participant.
    """

    # Remove duplicate match IDs
    match_ids = list(dict.fromkeys(match_ids))

    # Loop through each match ID to retrieve the match data
    match_data = pd.DataFrame()
    print("Número de match IDs:", len(match_ids))
    for match in match_ids:
        match_url = "".join(
            [
                "https://",
                region_extended,
                ".api.riotgames.com/tft/match/v1/matches/",
                match,
                "?api_key=",
                api_key,
            ]
        )
        match_resp = requests.get(match_url, timeout=127).json()
        flat_match_resp = flatten(match_resp)

        # Flatten the JSON response to handle nested dictionaries
        flat_match_resp = flatten(match_resp)

        # Loop through each of the 8 participants in the match
        for i in range(8):
            # Create the prefix for the current player's data keys
            key_prefix = f"info_participants_{i}_"

            # Extract and clean the current player's data by removing the prefix
            player_info = {
                k.replace(key_prefix, ""): v
                for k, v in flat_match_resp.items()
                if k.startswith(key_prefix)
            }
            # Convert the player's data to a DataFrame, adjust data types,
            # and exclude non-numeric columns
            player_df = (
                pd.DataFrame([player_info])
                .convert_dtypes()
                .select_dtypes(exclude=["object"])
            )

            # Append the player's DataFrame to the main match_data DataFrame
            match_data = pd.concat([match_data, player_df], ignore_index=True)

    return match_data


def pipeline_data_analysis(
    match_data: List[str] 
):
    """
    Applies a data analysis pipeline to the match data.

    The pipeline includes the following steps:
        - Drop missing values (NaNDropper).
        - Drop corrupted entries (CorruptedDropper).
        - Reset index (ResetIndex).
        - Describe missing values (DescribeMissing).

    Args:
        match_data (List[str]): A list of match data.

    Returns:
        pd.DataFrame: A DataFrame with the processed match data.
    """
    # use pipeline for data analysis
    pipe_analysis = Pipeline([
       ("nandrop", NaNDropper()),
       ("corruptdropper", CorruptedDropper()),
       ("resetindex", ResetIndex()),
       ("nanpercent", DescribeMissing())
    ])
    match_data = pipe_analysis.fit_transform(match_data)
    # write csv for data analysis
    match_data.to_parquet(
        'data/unprocessed_matched_data.parquet', 
        index = False
    )

    return match_data

def pipeline_ml(
    match_data: List[str] 
):
    """
    Applies a machine learning preprocessing pipeline 
    to the match data.

    The pipeline includes the following steps:
        - Drop certain columns used only for training (TrainDropper).
        - Remove outliers (OutlierRemover).
        - Generate dummy variables for categorical features (GetAugmentDummies).

    Args:
        match_data (List[str]): A list of match data.

    Returns:
        pd.DataFrame: A DataFrame with the processed match data ready for machine learning.
    """
    pipe_ml = Pipeline([
            ("name_dropper", TrainDropper()),
            ("outlier_dropper", OutlierRemover()),
            ("augmentdummies", GetAugmentDummies())
    ])

    match_data = pipe_ml.fit_transform(match_data)

    # write parquet file
    match_data.to_parquet(
        'data/processed_matched_data.parquet', 
        index = False
    )

    return match_data
