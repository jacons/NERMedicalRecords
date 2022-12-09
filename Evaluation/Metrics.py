# class EnsembleDataset(Dataset):
#   def __init__(self, dataset: DataFrame, conf: Configuration, parser: Parser):

#     self.__input_ids, self.__mask, self.__labelsA, self.__labelsB = [], [], [], []
#     tokenizer = AutoTokenizer.from_pretrained(conf.bert)

#     for _, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):
#       tokens, _labelsA, _labelsB = row[0].split(), row[2].split(), row[3].split()

#       # Apply the tokenization at each row
#       token_text = tokenizer(tokens, max_length=512, truncation=True, is_split_into_words=True,
#                              return_tensors="pt")

#       labelA_ids = parser.align_label(token_text.word_ids(), _labelsA)
#       labelB_ids = parser.align_label(token_text.word_ids(), _labelsB)

#       input_ = token_text['input_ids'].squeeze(0)
#       mask_ = token_text['attention_mask'].squeeze(0)

#       labelA_ = LongTensor(labelA_ids)
#       labelB_ = LongTensor(labelB_ids)

#       if conf.cuda:
#           input_ = input_.to("cuda:0")
#           mask_ = mask_.to("cuda:0")
#           labelA_ = labelA_.to("cuda:0")
#           labelB_ = labelB_.to("cuda:0")

#       self.__input_ids.append(input_)
#       self.__mask.append(mask_)
#       self.__labelsA.append(labelA_)
#       self.__labelsB.append(labelB_)

#   def __len__(self):
#       return len(self.__input_ids)

#   def __getitem__(self, idx):
#       return self.__input_ids[idx], self.__mask[idx], self.__labelsA[idx], self.__labelsB[idx]

# ts = DataLoader(EnsembleDataset(df_test,conf,parser), collate_fn=padding_batch2, batch_size=1)
# default_id = parser.labels_to_ids["O"]

# def merge_label(labelsA:Tensor,labelsB:Tensor):

#   labelsA,labelsB = labelsA.squeeze(), labelsB.squeeze()
#   labels = []
#   for a,b in zip(labelsA,labelsB):
#     a, b = a.item(), b.item()
#     if a == b or b == default_id:
#       labels.append(a)
#     elif a == default_id:
#       labels.append(b)
#     else:
#       labels.append([a,b]) 
#   return labels
# parser.labels("dict")
# modelA = BertModel(conf.bert, 9)
# modelA.load_state_dict(torch.load(conf.folder + "tmp/modelA1.pt",map_location=torch.device('cpu')))

# modelB = BertModel(conf.bert, 5)
# modelB.load_state_dict(torch.load(conf.folder + "tmp/modelG2.pt",map_location=torch.device('cpu')))
# c = 0
# for input_id, mask, labelsA, labelsB in ts:

#   logits = modelB(input_id, mask, None)
#   # print(input_id.shape)
#   # predictions = logits[0][labelsA[0] != -100].argmax(dim=1)
#   input_id = input_id.squeeze()
#   print(logits[0].squeeze().argmax(dim=1))
#   # labels = merge_label(labelsA,labelsB)
#   if c == 1:
#     break
#   else:
#     c += 1

# model = BertModel(conf.bert, parser.labels("num"))
# model.load_state_dict(torch.load(conf.folder + "tmp/modelA1.pt"))

# if conf.cuda:
#   model = model.to("cuda:0")

# # train(model, parser, df_train, df_val, conf)
# # # Evaluation single tag modelA e tagA, modelB e tagB
# # # ensembling modelA modelB tag A e b
# # # evaludation modelAB e tag AeB

# def single_eval(model:BertModel, parser:Parser, conf:Configuration, df:DataFrame):

#   ts = DataLoader(NerDataset(df, conf, parser), collate_fn=padding_batch, batch_size=1)

#   max_label = parser.labels("num")
#   default_id = parser.entity_handler.labels_to_ids["O"]

#   matrix_results = torch.zeros(size=(max_label, max_label))

#   iter_label = range(max_label)
#   accuracy: Tensor = torch.zeros(max_label)
#   precision: Tensor = torch.zeros(max_label)
#   recall: Tensor = torch.zeros(max_label)
#   f1: Tensor = torch.zeros(max_label)

#   loss_ts = 0
#   # ========== Testing Phase ==========
#   with no_grad():
#       for input_id, mask, ts_label in tqdm(ts):
#           loss, logits = model(input_id, mask, ts_label)
#           loss_ts += loss.item()

#           label_clean = ts_label[0][ts_label[0] != -100]
#           predictions = logits[0][ts_label[0] != -100].argmax(dim=1)

#           for lbl, pre in zip(label_clean, predictions):
#             if pre >= max_label:
#               pre = default_id
#             matrix_results[lbl, pre] += 1
#   # ========== Testing Phase ==========
#   loss_ts = loss_ts / len(ts)

#   for i in iter_label:
#       fn = torch.sum(matrix_results[i, :i]) + torch.sum(matrix_results[i, i + 1:])
#       fp = torch.sum(matrix_results[:i, i]) + torch.sum(matrix_results[i + 1:, i])
#       tn, tp = 0, matrix_results[i, i]
#       for x in iter_label:
#           for y in iter_label:
#               if (x != i) & (y != i):
#                   tn += matrix_results[x, y]
#       accuracy[i] = (tp + tn) / (tp + fn + fp + tn)
#       precision[i] = tp / (tp + fp)
#       recall[i] = tp / (tp + fn)
#       f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

#   print(loss_ts)
#   return {
#     "Accuracy": accuracy,
#     "Precision": precision,
#     "Recall": recall,
#     "F1 score": f1}
# single_eval(model,parser,conf,df_test)
