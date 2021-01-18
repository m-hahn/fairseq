import math
import sys
import torch
task = sys.argv[1]

assert task == "MRPC"

def mean(values):
   return sum(values)/len(values)

sensitivityHistogram = [0 for _ in range(40)]


from scipy.optimize import linprog


def getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities):
   #print(perSubsetSensitivities)
   c = [-x for x in perSubsetSensitivities]
   res = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds)
   # find the highly sensitive partition
   return -res.fun, res.x

from random import shuffle

alternatives_predictions_binary = {}
alternatives_predictions_float = {}
predictions_all = []


with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/MRPC/dev_datapoints_predictions_fairseq.tsv", "r") as inFile:
   itemsPredictions = dict([(x[0]+" "+x[1], x) for x in [x.split("\t") for x in inFile.read().strip().split("\n")]])

with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/MRPC/dev_alternatives_predictions_finetuned_RoBERTa.tsv", "r", encoding='utf-8') as inFile:
  for line in inFile:
     if len(line) < 5:
       continue
     line = line.strip().split("\t")
     if len(line) == 2:
       line.append("0.0")
     sentence1, sentence2, cont, binary = line
     sentence = sentence1.strip()+"@"+sentence2.strip()
     cont = float(cont)
     assert cont <= 0.0
     alternatives_predictions_binary[sentence.strip()] = 1 if binary.strip() == "entailment" else 0
     alternatives_predictions_float[sentence.strip()] = cont
     predictions_all.append(cont)
  print(len(alternatives_predictions_binary))

predictions_all = torch.FloatTensor(predictions_all)
variance_predictions = predictions_all.pow(2).mean(dim=0) - predictions_all.mean(dim=0).pow(2)
print(predictions_all)
print(predictions_all.mean(dim=0))
print(variance_predictions)
#quit()

with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/MRPC/dev_alternatives_c.tsv", "r", encoding='utf-8') as inFile:
  alternatives = inFile.read().strip().split("#####\n")
  print(len(alternatives))



from collections import defaultdict

RoBERTa_alternatives = defaultdict(list)
for group in [""]: # , "_d", "_e"
 with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/MRPC/dev_alternatives_RoBERTa_finetuned{group}.tsv", "r") as inFile:
  for line in inFile:
     line = line.strip().split("\t")
     if len(line) < 3:
           print("ERROR", line)
           continue
     RoBERTa_alternatives[(line[0].strip(), line[1].replace("</s>", "").strip())].append(line[2])



sensitivities = []


processed = set()

with open(f"/u/scr/mhahn/sensitivity/sensitivities/s1ensitivities_{__file__}", "w") as outFile:
 print("Original", "\t", "BinaryS1ensitivity", file=outFile)
 for alternative in alternatives:
   if len(alternative) < 5:
      continue
   variants_set = set()
   variants_dict = {}
   
   alternative = alternative.split("\n")
   original = alternative[0].replace("</s>", "").replace("@ ", "@").replace(" @", "@").strip()
#   print(list(itemsPredictions)[:10])
   print(original)
   print(original+"#")
#   assert original in itemsPredictions
#   entry = itemsPredictions[original]
#   predictionForOriginal = float(entry[2])
#   booleanPredictionForOriginal = 1 if (entry[3] == "entailment") else 0
#   assert predictionForOriginal <= 0
#   assert booleanPredictionForOriginal in [0,1]
   tokenized2 = alternative[2].replace("</s>", "").strip()
   tokenized = alternative[2].split(" ")
   valuesPerVariant = {}

   for variant in alternative[3:]:
      #print(variant)
      if len(variant) < 5:
         print("SHORT?", variant)
         continue
      try:
         subset, sentence= variant.strip().split("\t")
      except ValueError:
         print("ERROR", variant)
         continue
      subset = subset.strip()
      #print([(subset, tokenized2)])
      #print(list(BERT_alternatives)[:5])
      if ((subset, tokenized2) not in RoBERTa_alternatives):
         print("WEIRD", (subset, tokenized2))
         assert False
      if (subset, tokenized2) in processed:
        continue
      for alternative in RoBERTa_alternatives[(subset, tokenized2)]:
         #print(alternative)
         alternative = alternative.replace("<s>", "").replace("</s>", "").strip().replace("@ ", "@").replace(" @", "@")
         if alternative not in alternatives_predictions_float:
#            print("DID NOT FIND", alternative)
            ats = [x for x in alternative if x == "@"]
            #assert len(ats) > 1, "#"+alternative+"#"
#            assert False, "#"+alternative+"#"
            continue
         valuesPerVariant[alternative] = alternatives_predictions_float[alternative]
         if subset not in variants_dict:
            variants_dict[subset] = []
         variants_dict[subset].append(alternative)
  # print((result))

   varianceBySubset = {}
   for subset in variants_dict:
       values = torch.FloatTensor([ valuesPerVariant[x] for x in variants_dict[subset]]).exp()
       varianceBySubset[subset] = 4*float((values.mean(dim=0) - values).pow(2).mean(dim=0).max())
       assert varianceBySubset[subset] <= 1
   #    print(values)
  #     print(len(variants_dict[subset])) # WHY is this 100?
 #      print(varianceBySubset[subset])
#       assert False


   subsetsEnumeration = list(variants_dict)
   if len(subsetsEnumeration) == 0:
     continue 
   N = len(subsetsEnumeration[0])
   A = [[0 for subset in range(len(subsetsEnumeration))] for inp in range(N)]
   for inp in range(N):
       for subset, bitstr in enumerate(subsetsEnumeration):
          assert len(bitstr) == N
          if bitstr[inp] == "1":
              A[inp][subset] = 1
   
   
   b = [1 for _ in range(N)]
   x_bounds = [(0,1) for _ in range(len(subsetsEnumeration))]
   perSubsetSensitivities = [varianceBySubset[x] for x in subsetsEnumeration]
#   print(perSubsetSensitivities)
 #  quit()
   sensitivity, _ = getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities)
   print("OVERALL SENSITIVITY ON THIS DATAPOINT", sensitivity)
   try:
      sensitivityHistogram[int(2*sensitivity)] += 1
   except IndexError:
      print("Index Error")
   sensitivities.append(sensitivity)
   print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))
   print(original, "\t", sensitivity, file=outFile)

print("Examples", len(sensitivities))
print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))
print("Median block sensitivity of the model", sorted(sensitivities)[int(len(sensitivities)/2)])


sensitivityHistogram = torch.FloatTensor(sensitivityHistogram)
print(sensitivityHistogram/sensitivityHistogram.sum())


