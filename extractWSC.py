from examples.roberta.wsc import wsc_utils  # also loads WSC task and criterion
nsamples, ncorrect = 0, 0
with open("/u/scr/mhahn/PRETRAINED/WSC/val.txt", "w") as outFile:
  for sentence, label in wsc_utils.jsonl_iterator('/juicier/scr120/scr/mhahn/PRETRAINED/WSC/val.jsonl', eval=True):
     print(sentence, file=outFile)
    



