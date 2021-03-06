import math
import sys
import torch
task = sys.argv[1]

assert task == "SST-2"

def mean(values):
   return sum(values)/len(values)

sensitivityHistogram = [0 for _ in range(40)]



from scipy.optimize import linprog


def getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities):
   #print(perSubsetSensitivities)
   c = [-x for x in perSubsetSensitivities]
   res = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds)
   # find the highly sensitive partition
   return -res.fun

from random import shuffle

alternatives_predictions_binary = {}
alternatives_predictions_float = {}

averageLabel = [0,0,0]


with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_datapoints_predictions_fairseq.tsv", "r") as inFile:
   itemsPredictions = dict([(x[0], x) for x in [x.split("\t") for x in inFile.read().strip().split("\n")]])

with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_alternatives_predictions_fairseq.tsv", "r") as inFile:
  for line in inFile:
     if len(line) < 5:
       continue
     line = line.strip().split("\t")
     if len(line) == 2:
       line.append("0.0")
     sentence, prediction_log, prediction_discrete = line
     alternatives_predictions_float[sentence.strip()] = torch.FloatTensor([float(prediction_log)])
     averageLabel[0]+=1
     averageLabel[1]+=math.exp(float(prediction_log))
     averageLabel[2]+=(math.exp(float(prediction_log)))**2
  print(len(alternatives_predictions_float))

print("Average Label", 2*averageLabel[1]/averageLabel[0]-1)
print("Label Variance", 4*(averageLabel[2]/averageLabel[0] - (averageLabel[1]/averageLabel[0])**2))
#quit()

print(list(alternatives_predictions_float.items())[:10])

with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_alternatives_respectWordBoundaries.tsv", "r") as inFile:
  alternatives = inFile.read().strip().split("#####\n")
  print(len(alternatives))

sensitivities = []

with open(f"/u/scr/mhahn/sensitivity/sensitivities/s11ensitivities_{__file__}", "w") as outFile:
 print("Original", "\t", "BinaryS11ensitivity", file=outFile)
 for alternative in alternatives:
   if len(alternative) < 5:
      continue
   variants_set = set()
   variants_dict = {}
   
   alternative = alternative.split("\n")
   original = alternative[0].strip()
   print(original+"#")
   assert original in itemsPredictions
   entry = itemsPredictions[original]
   predictionForOriginal = float(entry[2])
   assert predictionForOriginal <= 0
   print(entry)
   tokenized = alternative[1].split(" ")
   for variant in alternative[3:]:
      #print(variant)
      if len(variant) < 5:
         continue

      subset, sentence= variant.strip().split("\t")
      sentence = "".join(sentence.strip().split(" "))
      sentence = sentence.replace("▁", " ")
      sentence = sentence.strip()
      if sentence not in alternatives_predictions_float:
         print("DID NOT FIND", sentence)
         assert False
         continue
      assert sentence in alternatives_predictions_float, sentence


      variants_set.add(sentence)
      if subset not in variants_dict:
         variants_dict[subset] = []
      variants_dict[subset].append(sentence)
  # print((result))
   print(len(variants_set), "variants")
   valuesPerVariant = {}
   for variant in variants_set:
   #  print(variant)
     try:
       valuesPerVariant[variant] = alternatives_predictions_float[variant]
     except ValueError:
        print("VALUE ERROR", variant)
        valuesPerVariant[variant] = 0
     except AttributeError:
        print("VALUE ERROR", variant)
        valuesPerVariant[variant] = 0


   varianceBySubset = {}
   for subset in variants_dict:
       values = torch.stack([ valuesPerVariant[x] for x in variants_dict[subset]], dim=0).exp()
       varianceBySubset[subset] = 4*float((values.mean(dim=0) - values).pow(2).mean(dim=0).max())
       assert varianceBySubset[subset] <= 1, varianceBySubset[subset]
#   print(varianceBySubset)


   subsetsEnumeration = sorted(list(variants_dict))
   if len(subsetsEnumeration) == 0:
     continue 
   subsetsEnumeration2 = [subsetsEnumeration[0]]
   #print(subsetsEnumeration)
   for i in range(1, len(subsetsEnumeration)):
     if subsetsEnumeration[i].index("1") == subsetsEnumeration[i-1].index("1"):
        continue
     subsetsEnumeration2.append(subsetsEnumeration[i])
   subsetsEnumeration = subsetsEnumeration2
   print(subsetsEnumeration)
   #quit()
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

   sensitivity = getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities)
   if str(sensitivity) == "nan":
      continue
   print("OVERALL SENSITIVITY ON THIS DATAPOINT", sensitivity)
   sensitivityHistogram[int(2*sensitivity)] += 1
   sensitivities.append(sensitivity)
   print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))
   print(original, "\t", sensitivity, file=outFile)

print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))
print("Median block sensitivity of the model", sorted(sensitivities)[int(len(sensitivities)/2)])

import torch
sensitivityHistogram = torch.FloatTensor(sensitivityHistogram)
print(sensitivityHistogram/sensitivityHistogram.sum())


