from typing import Tuple

from pandas import concat, DataFrame

import configuration
from Parser.parser_utils import EntityHandler, buildDataset


def Parser(conf: Configuration) -> EntityHandler:
    print("Building sentences\n" + "-" * 85)

    datasets, set_entities = [], set()
    for type_entity in conf.type_of_entity:  # iterate all type of entity es a, b or both
        dt, labels = buildDataset(type_entity, conf)
        datasets.append(dt)
        set_entities.update(labels)

    datasets = concat(datasets, axis=1)
    unified_datasets = datasets.loc[:, ~datasets.columns.duplicated()].drop_duplicates()

    return EntityHandler("", unified_datasets, set_entities)


def EnsembleParser(conf: Configuration) -> Tuple[dict, DataFrame]:
    print("Building sentences\n" + "-" * 85)

    entity_handlers: dict = {}
    datasets = []
    for e in conf.type_of_entity:  # iterate all type of entity es a, b or both
        dt, labels = buildDataset(e, conf)

        entity_handlers[e] = EntityHandler(e, dt, labels)
        datasets.append(dt)

    datasets = concat(datasets, axis=1)
    unified_datasets = datasets.loc[:, ~datasets.columns.duplicated()].drop_duplicates()
    return entity_handlers, unified_datasets
