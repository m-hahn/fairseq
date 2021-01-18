from fairseq.models.roberta import RobertaModel
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()


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

from collections import defaultdict

alternativesPerPar = defaultdict(list)

with open(f'/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_alternatives_PMLM_1billion_raw_Separately.tsv', 'r') as fin:
  try:
    while True:
        line = next(fin).strip()
        try:
           subset, original_tokenized, alternative = line.strip().split("\t")
        except ValueError:
           print("ValueError: ", line)
           continue
        alternative = alternative.split("[SEP]")
        assert len(alternative) >= 1, alternative
        assert len(alternative[0]) > 5, alternative
        alternativesPerPar[(subset.strip(), original_tokenized.strip())].append(alternative[0])
  except StopIteration:
     pass

import random
evaluatedPairs = set()
constructedMatchesDone = set()

with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_alternatives_predictions_PMLM_1billion_raw_Separately_Matches.tsv', "w") as outFileMatches:
 with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_alternatives_predictions_PMLM_1billion_raw_Separately.tsv', "w") as outFile:
  for group in ["_c"]:
    with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_alternatives{group}.tsv", "r") as inFile:
     for line in inFile:
        if line.startswith("####"):
           print("######", file=outFileMatches)
           print(next(inFile).strip(), file=outFileMatches)
           boundary = int(next(inFile).strip())
           tokenized = next(inFile).strip()
           print(boundary, file=outFileMatches)
           print(tokenized, file=outFileMatches)
           print("TOK", tokenized)
           line = next(inFile)
        if len(line) < 3:
         continue
        try:
           mask, sampled = line.strip().split("\t")
        except ValueError:
           continue
        mask = mask.strip()
        key = (tokenized.strip(), mask.strip())
        if key in constructedMatchesDone:
           continue
        constructedMatchesDone.add(key)
    #    print(mask, tokenized, boundary)
        tokenized_ = tokenized.strip().split(" ")
        assert len(tokenized_) == len(mask)
   #     print(tokenized_[:boundary], tokenized_[boundary:])
  #      print(mask[:boundary]+"_SIDE_0")
 #       print(mask[boundary:]+"_SIDE_1")
#        print(list(alternativesPerPar)[:10])
        if "1" in mask[:boundary]:
            left = alternativesPerPar[(mask[:boundary]+"_SIDE_0", tokenized.strip())]
        else:
            leftPart = "".join(tokenized_[:boundary]).replace("▁", " ")
            left = [leftPart for _ in range(10)]
        if "1" in mask[boundary:]:
            right = alternativesPerPar[(mask[boundary:]+"_SIDE_1", tokenized.strip())]
        else:
            rightPart = "".join(tokenized_[boundary:]).replace("▁", " ")
            right = [rightPart for _ in range(10)]

  #      print(mask[:boundary], mask[boundary:])
   #     print(len(left), len(right))
        numSamples = min(len(left), len(right))
        for l, r in zip(random.sample(left, numSamples), random.sample(right, numSamples)):
             pair = [l, r]
#             print(pair)        
             for i in range(2):
                pair[i] = pair[i].replace("[CLS]", "").replace("[SEP]", "").strip().replace(" ' s ", " 's ").replace(" ' ll ", " 'll ").replace(" ' d ", " 'd ").replace("n ' t ", "n't ").replace(" ' ve ", " 've ").replace(" @ - @ ", "-").replace("( ", "(")
                pair[i] = detokenizer.detokenize(pair[i].split(" "))

             print(mask, "\t", pair[0], "\t", pair[1], file=outFileMatches)
             if tuple(pair) not in evaluatedPairs:
                 evaluatedPairs.add(tuple(pair))
             else:
                 continue

             if len(evaluatedPairs) % 100 == 0:
                print(len(evaluatedPairs), pair)
             tokens = roberta.encode(pair[0], pair[1])
             prediction = roberta.predict('sentence_classification_head', tokens)
             prediction_label = label_fn(prediction.argmax().item())
             prediction = [float(x) for x in prediction.view(-1)]
             print("\t".join([pair[0], pair[1], str(prediction[1]), {"not_entailment" : "0", "entailment" : "1"}[prediction_label]]), file=outFile)

