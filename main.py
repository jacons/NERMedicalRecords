from Configuration import Configuration
from Parser.Parsers import Parser
from Parser.parser_utils import Splitting

if __name__ == "__main__":

    conf = Configuration()
    entity_handler = Parser(conf)

    df_train, df_val, df_test = Splitting().holdout(entity_handler.get_sentences())

    """
    model = BertModel(conf.bert, parser.labels("num"))

    if conf.cuda:
        model.to("cuda:0")

    train(model, parser, df_train, df_val, conf) 
   """
