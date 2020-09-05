"""
Evaluate average block sensitivity for WSC.
"""
import traceback

from collections import defaultdict

fromOriginal = {}
with open("/u/scr/mhahn/PRETRAINED/WSC/val_alternatives_c.txt", "r") as inFile: # _SECOND_VERSION_BACKUP
  alternatives = [x.split("\n") for x in inFile.read().strip().split("#####\n")]
for alt in alternatives:
  if len(alt) < 3:
      continue
  original = alt[0].strip()
  boundaries = [int(x) for x in alt[1].strip().split(" ")]
  tokenized = alt[2].strip().split(" ")
  boundaries_ = boundaries
  if boundaries_[0] > boundaries_[2]:
    boundaries_ = boundaries_[2:] + boundaries[:2]
  tokenized.insert(boundaries_[3], "]")
  tokenized.insert(boundaries_[2], "[")
  tokenized.insert(boundaries_[1], "_")
  tokenized.insert(boundaries_[0], "_")
  tokenized = " ".join(tokenized)
  fromOriginal[tokenized] = (original, boundaries)

print(len(fromOriginal))
print(len(alternatives))

from fairseq.models.roberta import RobertaModel
from examples.roberta.wsc import wsc_utils  # also loads WSC task and criterion
from examples.roberta.wsc import wsc_task
#roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'WSC/')
nsamples, ncorrect = 0, 0

def mean(values):
   return sum(values)/len(values)

sensitivityHistogram = [0 for _ in range(40)]


def variance(values):
   return mean([x**2 for x in values]) - mean(values)**2

roberta = RobertaModel.from_pretrained('/u/scr/mhahn/PRETRAINED/roberta.large.wsc', "model.pt", "/u/scr/mhahn/PRETRAINED/WSC/")
roberta.cuda()



from random import shuffle

evaluatedSoFar = set()

hasEvaluated = set()

with open(f"/u/scr/mhahn/PRETRAINED/WSC/val_alternatives_predictions_raw_RoBERTa.txt", "w") as outFile:
 with open("/u/scr/mhahn/PRETRAINED/WSC/val_alternatives_RoBERTa_raw.tsv", "r") as inFile: # _SECOND_VERSION_BACKUP
  while True:
   try:
      line = next(inFile).strip()
   except StopIteration:
      print("File end", 62)
      break
   try:
      subset, original_tokenized, alternative = line.strip().split("\t")
   except ValueError:
      print("ValueError: ", line)
      continue
   if len(alternative) < 5:
      continue
   variants_set = set()
   variants_dict = {}
   original_tokenized_list=original_tokenized.strip().split(" ")
   toNPs = {}
   alternative = alternative.replace("<s>", "").replace("</s>", "").strip()
   #print(original_tokenized)
   assert (original_tokenized.strip() in fromOriginal), original_tokenized
   _, boundaries = (fromOriginal[original_tokenized.strip()])
   if alternative in hasEvaluated:
      continue
   hasEvaluated.add(alternative)
   if True: 
     variant=alternative
     variant = alternative.replace(" [ ", " [").replace(" _ ", " _")
     if variant.startswith("_") and variant[1] == " ":
       variant = "_"+variant[2:]
     assert " _ " not in " "+variant+" "
     #print(variant)
     if boundaries[0] > boundaries[2]:
        if "#" in variant:
          print("# symbol occurs", variant)
          continue
        assert "#" not in variant, variant
        variant = variant.replace("_", "#")
        variant = variant.replace("[", "_")
        variant = variant.replace("]", "_")
        ind1 = variant.index("#")
        variant = variant[:ind1]+"["+variant[ind1+1:]
        ind1 = variant.index("#")
        variant = variant[:ind1]+"]"+variant[ind1+1:]
     if "] " not in variant: # Occurs occasionally: ]'re
        variant = variant.replace("]", "] ")
     try:
       value = 1 if roberta.disambiguate_pronoun(variant) == True else -1
       print(alternative, "\t", value, file=outFile)
       if len(hasEvaluated) % 100 == 0:
         print(variant, len(hasEvaluated))
     except ValueError:
        print("VALUE ERROR", variant)
#        valuesPerVariant[variant] = 0
     except AttributeError:
        print("ATTRIBUTE ERROR", variant)
 #       valuesPerVariant[variant] = 0
     except RuntimeError:
        print("RUNTIME ERROR", variant)
  #      valuesPerVariant[variant] = 0
     except Exception as e:
        print("EXCEPTION ", variant)
        print(traceback.print_exc())
        print(e)
#        assert False
   #     valuesPerVariant[variant] = 0

