from Configuration import Configuration
from Parsers.Parser import Parser

if __name__ == "__main__":

    conf = Configuration()
    parser = Parser(conf)

    """
    df_train, df_val, df_test = Splitting().holdout(parser.get_sentences(), size=1)
    model = BertModel(conf.bert, parser.labels("num"))

    if conf.cuda:
        model.to("cuda:0")

    train(model, parser, df_train, df_val, conf) 
   """
