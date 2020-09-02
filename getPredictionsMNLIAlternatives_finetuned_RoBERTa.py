# Based on the script provided in https://github.com/pytorch/hub/blob/master/pytorch_fairseq_roberta.md

from fairseq.models.roberta import RobertaModel

import torch
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')

import torch
label_fn = lambda label: roberta.task.label_dictionary.string(
    torch.LongTensor([label + roberta.task.label_dictionary.nspecial])
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
evaluatedSoFar = set()
lineNumbers = 0
with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/MNLI/dev_alternatives_c_predictions_finetuned_RoBERTa.tsv', "w") as outFile:
 for group in ["", "_d", "_e"]:
  try:
   with open(f'/u/scr/mhahn/PRETRAINED/GLUE/glue_data/MNLI/dev_alternatives_RoBERTa_finetuned{group}.tsv', 'r') as fin:
    while True:
        lineNumbers += 1
        line = next(fin).strip()
        try:
           subset, original_tokenized, alternative = line.strip().split("\t")
        except ValueError:
           print("ValueError: ", line)
           continue
                                                                                                         
        alternative = alternative.replace("<s>", "").replace("</s>", "").strip().replace("@ @", "@")
        _sAt = [i for  i in range(len(alternative)) if alternative[i] == "@" and i > 0]
        if  len(_sAt) != 1:
              print("ERROR", alternative)
              continue
        
        sentences = [alternative[:_sAt[0]].strip(), alternative[_sAt[0]+1:].strip()]
        if tuple(sentences) in evaluatedSoFar:
           continue
        evaluatedSoFar.add(tuple(sentences))
        if len(evaluatedSoFar) % 100 == 0:
           print(len(evaluatedSoFar), sentences)
        tokens = roberta.encode(sentences[0], sentences[1])
        prediction = roberta.predict('mnli', tokens)
        prediction_label = prediction.argmax().item()
        prediction = [float(x) for x in prediction.view(-1)]
        print("\t".join([sentences[0], sentences[1], str(prediction_label), " ".join([str(x) for x in prediction])]), file=outFile)
  except StopIteration:
     pass
  except FileNotFoundError:
     pass

