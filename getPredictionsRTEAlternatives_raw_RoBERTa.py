from fairseq.models.roberta import RobertaModel


import sys

model = sys.argv[1]
assert model == "RTE"

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
with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_alternatives_predictions_raw_RoBERTa.tsv', "w") as outFile:
 for group in ["", "_d", "_e"]:
  try:
   with open(f'/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_alternatives_RoBERTa_raw{group}.tsv', 'r') as fin:
    while True:
        line = next(fin).strip()
        try:
           subset, original_tokenized, alternative = line.strip().split("\t")
        except ValueError:
           print("ValueError: ", line)
           continue

        alternativeOriginal = alternative.strip()
                                                                                                         
        alternative = alternative.replace("<s>", "").replace("</s>", "").strip().replace("@ @", "@")
        _sAt = [i for  i in range(len(alternative)) if alternative[i] == "@" and i > 0]
        if  len(_sAt) != 1:
              print("ERROR", alternative)
              continue
        
        sentences = [alternative[:_sAt[0]].strip(), alternative[_sAt[0]+1:].strip()]
        if alternativeOriginal in evaluatedSoFar:
           continue
        evaluatedSoFar.add(alternativeOriginal)
        if len(evaluatedSoFar) % 100 == 0:
           print(len(evaluatedSoFar), sentences)
        tokens = roberta.encode(sentences[0], sentences[1])
        prediction = roberta.predict('sentence_classification_head', tokens)
        prediction_label = label_fn(prediction.argmax().item())
        prediction = [float(x) for x in prediction.view(-1)]
        print("\t".join([alternativeOriginal, str(prediction[1]), {"not_entailment" : "0", "entailment" : "1"}[prediction_label]]), file=outFile)
  except StopIteration:
     pass    
