import numpy as np
import pandas as pd
import requests
from flatten_json import flatten
### data pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import json
from typing import Optional

def get_api_key(json_file_path: str) -> Optional[str]:
    """
    Reads an API key from a JSON file.

    Parameters:
    - json_file_path (str): The path to the JSON file.

    Returns:
    - Optional[str]: The API key retrieved from the JSON file, or None if not found.
    """
    try:
        # Load the JSON file
        with open(json_file_path, 'r') as file:
            config = json.load(file)

        # Access the API key
        api_key = config.get('api_key')

        return api_key
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def load_config_from_json(json_file_path):
    """
    Load configuration from a JSON file.

    Parameters:
    - json_file_path: str, path to the JSON configuration file.

    Returns:
    - api_key: str, Riot API key.
    - REGION: str, region code.
    - REGION_EXTENDED: str, extended region name.
    - N_MATCHES: int, number of matches to retrieve.
    """
    # Load the JSON data from the file
    with open(json_file_path, 'r') as file:
        config = json.load(file)

    # Assign the values from the JSON to your variables
    region = config.get('REGION', 'euw1') 
    region_extended = config.get('REGION_EXTENDED', 'Europe')
    n_matches = config.get('N_MATCHES', 20) 

    return region, region_extended, n_matches

def get_challengers(api_key: 'str', region: 'str') -> 'json':

    ph_gm_url = ''.join([
        'https://',
        region,
        '.api.riotgames.com/tft/league/v1/challenger?queue=RANKED_TFT'
    ])
    ph_gm_url = ph_gm_url + '&api_key=' + api_key

    try:
        ph_gm_resp = requests.get(ph_gm_url, timeout = 127)
        gm_info = ph_gm_resp.json()
        return gm_info
    except:  # noqa: E722
        print('Request has timed out.')

def get_id(players: 'json') -> list:

    player_ids = [player['summonerId'] for player in players['entries']]
    return player_ids

def get_puuid(names: list, region:str) -> list:

    puuids = []
    for name in names:
        puuid_url = ''.join([
            'https://',
            region,
            '.api.riotgames.com/tft/summoner/v1/summoners/'
        ])
        puuid_url = puuid_url + name + '?api_key=' + api_key
        puuid_resp = requests.get(puuid_url, timeout = 127)
        puuids += [dict(puuid_resp.json())]

    final_puuids = []
    for i in range(len(puuids)):
        final_puuids += [puuids[i].get('puuid')]

    return final_puuids

def get_match_ids(puuids: list, region_extended: str, n_matches: int) -> list:
    '''
    Get the last matches of players 
    '''
    match_ids = []
    for puuid in puuids:
        match_url = ''.join([
            'https://',
            region_extended,
            '.api.riotgames.com/tft/match/v1/matches/by-puuid/'
        ])
        match_url += ''.join([
            puuid,
            '/ids?start=0&count=',
            str(n_matches),
            '&api_key=',
            api_key
        ])
        match_resp = requests.get(match_url, timeout = 127)
        match_ids += match_resp.json()

    return match_ids

def get_match_data(match_ids: list, region_extended: str) -> 'pd.DataFrame':

    match_data = pd.DataFrame()
    print("Número de match IDs:", len(match_ids))
    for match in match_ids: 
        match_url = ''.join([
            'https://',
            region_extended,
            '.api.riotgames.com/tft/match/v1/matches/',
            match,
            '?api_key=',
            api_key
        ])
        match_resp = requests.get(match_url, timeout = 127).json()
        flat_match_resp = flatten(match_resp)

        # Flatten the JSON response to handle nested dictionaries
        flat_match_resp = flatten(match_resp)
        
        # Loop through each of the 8 participants in the match
        for i in range(8):
            # Create the prefix for the current player's data keys
            key_prefix = f'info_participants_{i}_'

            # Extract and clean the current player's data by removing the prefix
            player_info = {
                k.replace(key_prefix, ''): v 
                for k, v in flat_match_resp.items() 
                if k.startswith(key_prefix)
            }            
            # Convert the player's data to a DataFrame, adjust data types, 
            # and exclude non-numeric columns
            player_df = pd.DataFrame([player_info]).convert_dtypes()\
                .select_dtypes(exclude=['object'])
            
            # Append the player's DataFrame to the main match_data DataFrame
            match_data = pd.concat([match_data, player_df], ignore_index=True)

    return match_data

    
class NaNDropper(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return X.dropna(how = 'all').dropna(axis = 'columns', how = 'all')
    
class CorruptedDropper(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        corrupted_features = ['units_5_items_0', 'units_5_items_1',	
                            'units_5_items_2', 'units_6_items_0',
                            'units_6_items_1', 'units_6_items_2',
                            'units_7_items_0', 'units_7_items_1',	
                            'units_7_items_2', 'units_3_items_0',
                            'units_3_items_1', 'units_0_items_0',
                            'units_1_items_0', 'units_1_items_1',	
                            'units_2_items_0', 'units_2_items_1',	
                            'units_2_items_2', 'units_1_items_2',
                            'units_4_items_0', 'units_4_items_1',	
                            'units_4_items_2', 'units_0_items_1',	
                            'units_3_items_2', 'units_0_items_2',	
                            'units_8_items_0', 'units_8_items_1',	
                            'units_8_items_2']
        X = X.drop(corrupted_features, axis='columns', errors='ignore')

        return X
        
class ResetIndex(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return X.reset_index(drop = True)

class DescribeMissing(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        # get number of missing data points per column
        missing_values_count = X.isnull().sum()

        # how many missing values do we have?
        total_cells = np.prod(X.shape)
        total_missing = missing_values_count.sum()

        # percent of missing data
        percent_missing = (total_missing / total_cells) * 100
        print('Percent Missing of Data: ' + str(percent_missing))

        return X

# Initialize first pipe

pipe_analysis = Pipeline([
       ("nandrop", NaNDropper()),
       ("corruptdropper", CorruptedDropper()),
       ("resetindex", ResetIndex()),
       ("nanpercent", DescribeMissing())
])

### Data Pipeline for ML
class TrainDropper(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        # remove features that don't help with training the data
        non_training_features = ['companion_content_ID', 'companion_item_ID',
                                'companion_skin_ID', 'companion_species',
                                'gold_left', 'players_eliminated']
        
        for feature in non_training_features:
            try:
                X = X.drop(feature, axis = 'columns')
            except:  # noqa: E722
                continue
        
        return X
    
class OutlierRemover(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        # remove outliers (10% threshold to not remove level 8 data)
        threshold = int(len(X) * 0.1)
        X = X.dropna(axis = 1, thresh = threshold)
        
        return X

class GetAugmentDummies(BaseEstimator, TransformerMixin):

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        augments = ['augments_0', 'augments_1', 'augments_2']
        X = pd.get_dummies(X, columns = augments)

        return X

# Initialize ML Pipeline

pipe_ml = Pipeline([
        ("name_dropper", TrainDropper()),
        ("outlier_dropper", OutlierRemover()),
        ("augmentdummies", GetAugmentDummies())
])

def use_data_pipeline(match_data: 'json', filename: str) -> 'pd.DataFrame':

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
        'data/unprocessed_' + filename + '.parquet', 
        index = False
    )

    pipe_ml = Pipeline([
            ("name_dropper", TrainDropper()),
            ("outlier_dropper", OutlierRemover()),
            ("augmentdummies", GetAugmentDummies())
    ])

    match_data = pipe_ml.fit_transform(match_data)

    # write csv for placement estimator
    match_data.to_parquet(
        'data/processed_' + filename + '.parquet', 
        index = False
    )

    return match_data

if __name__ == "__main__":

    try:
        config_path = './data/config.JSON' 
        api_path = './data/API_key.json'
        region, region_extended, n_matches = load_config_from_json(config_path)
        print("Getting API Key...")
        api_key = api_key = get_api_key('./data/API_key.json')
        print("Getting master data...")
        challengers = get_challengers(api_key, region)
        print("Getting names of the challengers players...")
        players_names = get_id(challengers)
        print("Getting puuids of the challengers players...")
        players_puuids = get_puuid(players_names, region)
        print("Getting match_ids of the challengers players...")
        matches = get_match_ids(
            players_puuids, 
            region_extended, 
            n_matches=n_matches
        )
        # remove duplicate matches
        matches = list(dict.fromkeys(matches))
        # TMP
        print("Getting match data of the matches...")
        matches_data = get_match_data(matches, region_extended)
        print("USING PIPELINE...")
        processed_chall_match_data = use_data_pipeline(
            matches_data, 
            'matches_data'
        )

    except Exception as error: 
        print(error)
