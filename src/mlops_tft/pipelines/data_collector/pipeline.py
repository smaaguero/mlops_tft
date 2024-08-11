"""
This is a boilerplate pipeline 'data_collector'
generated using Kedro 0.19.7
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import get_challenger_data, get_id, get_puuid, get_match_ids,  get_match_data, pipeline_data_analysis, pipeline_ml


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            name='get_challenger_data',
            func=get_challenger_data,
            inputs=[
                'params:API_key',
                'params:region'
            ],
            outputs="get_challenger_data"
        ),
        node(
            name='get_id',
            func=get_id, # noqa: F405
            inputs=[
                'get_challenger_data'
            ],
            outputs="get_id"
        ),
        node(
            name='get_puuid',
            func=get_puuid, # noqa: F405
            inputs=[
                'get_id',
                'params:region',
                'params:API_key'
            ],
            outputs="get_puuid"
        ),
        node(
            name='get_match_ids',
            func=get_match_ids,  # noqa: F405
            inputs=[
                'get_puuid',
                'params:region_extended',
                'params:n_matches',
                'params:API_key'
            ],
            outputs="get_match_ids"
        ),
        node(
            name='get_match_data',
            func=get_match_data,
            inputs=[
                'get_match_ids',
                'params:region_extended',
                'params:API_key'
            ],
            outputs="get_match_data"
        ),
        node(
            name='pipeline_data_analysis',
            func=pipeline_data_analysis,
            inputs=[
                'get_match_data',
            ],
            outputs="pipeline_data_analysis"
        ),
        node(
            name='pipeline_ml',
            func=pipeline_ml,
            inputs=[
                'get_match_data',
            ],
            outputs="pipeline_ml"
        ),
    ])
