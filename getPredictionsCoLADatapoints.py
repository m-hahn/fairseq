from fairseq.models.roberta import RobertaModel


import sys

model = sys.argv[1]
assert model == "CoLA"

roberta = RobertaModel.from_pretrained(
    f'checkpoints_{model}/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path=f'{model}-bin'
)

import torch
label_fn = lambda label: roberta.task.label_dictionary.string(
    torch.LongTensor([label + roberta.task.label_dictionary.nspecial])
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
evaluatedSoFar = set()
with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/CoLA/dev_alternatives_c_finetuned.tsv') as fin:
  with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/CoLA/dev_datapoints_predictions_fairseq.tsv', "w") as outFile:
    while True:
        line = next(fin).strip()
        if line == "#####":
           original = next(fin).strip() # the original
           tokenized = next(fin) # the tokenized version
        else:
           continue
        sentences = [tokenized.strip().split(" ")]
        for i in range(1):
          sentences[i] = "".join(sentences[i])
          sentences[i] = sentences[i].replace("▁", " ").replace("</s>", "")
          sentences[i] = sentences[i].strip()
#        print(sentences)
        if tuple(sentences) in evaluatedSoFar:
           continue
        evaluatedSoFar.add(tuple(sentences))
        if len(evaluatedSoFar) % 100 == 0:
           print(sentences)
        tokens = roberta.encode(sentences[0])
        prediction = roberta.predict('sentence_classification_head', tokens)
        prediction_label = label_fn(prediction.argmax().item())
        prediction = [float(x) for x in prediction.view(-1)]
        print("\t".join([original, sentences[0], str(prediction[1]), prediction_label]), file=outFile)

