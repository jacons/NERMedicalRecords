from pandas import concat

from Configuration import Configuration
from Parsers.parser_utils import buildDataset, EntityHandler


class Parser:
    def __init__(self, conf: Configuration):

        print("Building sentences\n" + "-" * 85)

        datasets, set_entities = [], set()
        for type_entity in conf.type_of_entity:  # iterate all type of entity es a, b or both
            dt, labels = buildDataset(type_entity, conf)
            datasets.append(dt)
            set_entities.update(labels)

        datasets = concat(datasets, axis=1)
        datasets = datasets.loc[:, ~datasets.columns.duplicated()].drop_duplicates()

        self.entity_handler = EntityHandler("", datasets, set_entities)

    def align(self, token: list, labels: list):
        return self.entity_handler.align_label(token, labels)

    def get_sentences(self):
        return self.entity_handler.dt

    def labels(self, typ: str):
        if typ == "num":
            return len(self.entity_handler.set_entities)
        elif typ == "dict":
            return self.entity_handler.ids_to_labels
