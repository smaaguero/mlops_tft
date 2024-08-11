"""
This is a boilerplate pipeline 'place_predictor'
generated using Kedro 0.19.7
"""


from kedro.pipeline import Pipeline, pipeline, node
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            name='filter_columns',
            func=filter_columns,
            inputs=[
                'pipeline_ml',
            ],
            outputs="filter_columns"
        ),
        node(
            name='convert_dtypes',
            func=convert_dtypes,
            inputs=[
                'filter_columns',
            ],
            outputs="convert_dtypes"
        ),
        node(
            name='create_total_item_feature',
            func=create_total_item_feature,
            inputs=[
                'convert_dtypes',
            ],
            outputs="create_total_item_feature"
        ),
        node(
            name='fill_missing_values',
            func=fill_missing_values,
            inputs=[
                'create_total_item_feature',
            ],
            outputs="fill_missing_values"
        ),
        node(
            name='one_hot_encode',
            func=one_hot_encode,
            inputs=[
                'fill_missing_values',
            ],
            outputs="one_hot_encode"
        ),
        node(
            name='prepare_sets_data',
            func=prepare_sets_data,
            inputs=[
                'one_hot_encode',
                'params:test_size',
                'params:random_state'
            ],
            outputs=["X_train", "X_test", "y_train", "y_test"]
        ),
        node(
            name='train_models',
            func=train_models,
            inputs=[
                'X_train',
                'y_train',
                'X_test',
                'y_test'
            ],
            outputs=None
        ),
    ])
