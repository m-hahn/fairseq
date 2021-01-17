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
        alternativesPerPar[(subset.strip(), original_tokenized.strip())].append(alternative)
  except StopIteration:
     pass

constructedMatchesDone = set()

with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_alternatives_predictions_PMLM_1billion_raw_Separately.tsv', "w") as outFile:
 for group in ["_c"]:
  try:
   with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_alternatives{group}.tsv", "r") as inFile:
    for line in inFile:
        if line.startswith("####"):
           next(inFile)
           boundary = int(next(inFile).strip())
           tokenized = next(inFile).strip()
           print("TOK", tokenized)
           line = next(inFile)
        if len(line) < 3:
         continue
        try:
           mask, sampled = line.strip().split("\t")
        except ValueError:
           continue
        print(mask, tokenized, boundary)
        assert False, "Not yet implemented"
        quit()
        sampled = sampled.strip().split(" ")
        assert len(sampled) == len(tokenized.split(" ")), (sampled, tokenized)
        mask = mask.strip()
        assert len(sampled) == len(mask), (sampled, mask)
        masked = [sampled[i] if mask[i] == "0" else "[MASK]" for i in range(len(mask))]
        masked_pair = [masked[:boundary], masked[boundary:]] # "▁[CLS]"
        masks_pair = [mask[:boundary], mask[boundary:]]
        for side, masked, mask_side in zip(range(2), masked_pair, masks_pair):
          assert len(masked) == len(mask_side)
          #print(masked)
          masked = "".join(masked).replace("▁", " ").replace("[MASK]", " [MASK] ").replace("  ", " ").replace("</s>", "").strip()
          if "[MASK]" not in masked:
             continue
          #print(("CANDIDATE", (tokenized, mask, masked)))
          encodedWithMask = demo.encodeInputWithMask(masked, withCaching=True)
   #       lengthOfFirstPartPMLM = encodedWithMask.index(102)
    #      encodedWithMask = encodedWithMask[:lengthOfFirstPartPMLM] + encodedWithMask[lengthOfFirstPartPMLM+1:]
          maskString = "".join(["0" if x != 103 else "1" for x in encodedWithMask])
          key = (tokenized, mask_side+"_SIDE_"+str(side))
          if sideCache[key] > 10:
            continue
          sideCache[key] += 1
          blankCandidates.append({"tokenized" : tokenized, "XLNET_Mask" : mask_side+"_SIDE_"+str(side), "masked" : masked, "PMLM_Encoded" : encodedWithMask, "PMLM_Mask_Encoded" : maskString}) #, "lengthOfFirstPartPMLM" : lengthOfFirstPartPMLM})
          #print(blankCandidates[-1])
          if len(blankCandidates) % 1000 == 0:
             #break
             print("Recording blank candidates", len(blankCandidates))



        alternativeOriginal = alternative.strip()

        alternatives = alternative.replace("[CLS]", "").replace("[ CLS]", "").replace("[ CLS ]", "").split("[SEP]")
        assert len(alternatives) > 1, alternatives
        if len(alternatives) > 3 or (len(alternatives) > 2 and len(alternatives[2]) > 5):
            print("ODD Text after the end:", alternatives)
        alternatives = alternatives[:2]
        for i in range(2):
           alternatives[i] = alternatives[i].replace("[CLS]", "").replace("[SEP]", "").strip().replace(" ' s ", " 's ").replace(" ' ll ", " 'll ").replace(" ' d ", " 'd ").replace("n ' t ", "n't ").replace(" ' ve ", " 've ").replace(" @ - @ ", "-").replace("( ", "(")
           alternatives[i] = detokenizer.detokenize(alternatives[i].split(" "))

                                                                                                         
        
        sentences = alternatives
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
  except EOFError:
     pass
