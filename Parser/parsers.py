from typing import Tuple
from pandas import concat, DataFrame
from Parser.parser_utils import EntityHandler, buildDataset
from configuration import Configuration


def Parser(conf: Configuration, verbose=True) -> EntityHandler:
    """
    This is the main parser, it is used to build a dataset, it returns an instance of a EntityHandler
    which manage the sentences and the entities. The dataset created is defined in the configuration class.
    The final dataset will have one column for each group of entity selected. If more group of entities is used
    there will be merged together in a one big set of entities.

    This paser should be used for:
    1) Training
    2) Evaluate model with single or multiple groups of entity but not in "Ensemble mode"
    """
    if verbose:
        print("Building sentences\n" + "-" * 85)

    datasets, set_entities = [], set()
    for type_entity in conf.type_of_entity:  # iterate all type of entity es a, b or both
        dt, labels = buildDataset(type_entity, conf, verbose)
        datasets.append(dt)
        set_entities.update(labels)

    datasets = concat(datasets, axis=1)
    unified_datasets = datasets.loc[:, ~datasets.columns.duplicated()].drop_duplicates()

    return EntityHandler("", unified_datasets, set_entities)


def EnsembleParser(conf: Configuration, verbose=True) -> Tuple[dict, DataFrame]:
    """
    The EnsembleParser is equal to Standard Paser, but here the group of entities is treated separately.
    In fact, the method returns a dictionary of Handler which manage only a one group of entities.
    It also returns a unified dataset as before.
    The EnsembleParser should be user to evaluate model in "Ensemble mode".
    """
    if verbose:
        print("Building sentences\n" + "-" * 85)

    entity_handlers: dict = {}
    datasets = []
    for group_name in conf.type_of_entity:  # iterate all type of entity es a, b or both
        dt, labels = buildDataset(group_name, conf, verbose)

        # Create a handler for each group
        entity_handlers[group_name] = EntityHandler(group_name, dt, labels)
        datasets.append(dt)

    datasets = concat(datasets, axis=1)
    unified_datasets = datasets.loc[:, ~datasets.columns.duplicated()].drop_duplicates()
    return entity_handlers, unified_datasets
