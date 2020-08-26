from fairseq.models.roberta import RobertaModel


import sys

model = sys.argv[1]
assert model == "SST-2"

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
with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_alternatives_predictions_finetuned_ILM.tsv', "w") as outFile:
 for group in ["", "_d", "_e"]:
  try:
   with open(f'/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_alternatives_ILM{group}.tsv', 'r') as fin:
    while True:
        line = next(fin).strip()
        try:
           subset, original_tokenized, alternative = line.strip().split("\t")
        except ValueError:
           print("ValueError: ", line)
           continue
        alternative = alternative.replace("[CLS]", "").replace("[SEP]", "").strip()
        sentences = [alternative]
        if tuple(sentences) in evaluatedSoFar:
           continue
        evaluatedSoFar.add(tuple(sentences))
        if len(evaluatedSoFar) % 100 == 0:
           print(sentences)
        tokens = roberta.encode(sentences[0])
        prediction = roberta.predict('sentence_classification_head', tokens)
        prediction_label = label_fn(prediction.argmax().item())
        prediction = [float(x) for x in prediction.view(-1)]
        print("\t".join([sentences[0], str(prediction[1]), prediction_label]), file=outFile)
  except StopIteration:
     pass    
