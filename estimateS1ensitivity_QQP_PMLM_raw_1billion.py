import math
import sys
import torch
task = sys.argv[1]

assert task == "QQP"

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
predictions_all = []


with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/QQP/dev_alternatives_predictions_PMLM_1billion_raw.tsv", "r", encoding='utf-8') as inFile:
  for line in inFile:
     if len(line) < 5:
       continue
     line = line.strip().split("\t")
     if len(line) == 2:
       line.append("0.0")
     sentence, cont, binary = line
     cont = float(cont)
     assert cont <= 0.0
     alternatives_predictions_binary[sentence.strip()] = int(binary.strip())
     alternatives_predictions_float[sentence.strip()] = cont
     predictions_all.append(cont)
  print(len(alternatives_predictions_binary))

predictions_all = torch.FloatTensor(predictions_all)
variance_predictions = predictions_all.pow(2).mean(dim=0) - predictions_all.mean(dim=0).pow(2)
print(predictions_all)
print(predictions_all.mean(dim=0))
print(variance_predictions)
#quit()

with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/QQP/dev_alternatives_c.tsv", "r", encoding='utf-8') as inFile:
  alternatives = inFile.read().strip().split("#####\n")
  print(len(alternatives))



from collections import defaultdict

RoBERTa_alternatives = defaultdict(list)
for group in [""]: # , "_d", "_e"
 with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/QQP/dev_alternatives_PMLM_1billion_raw.tsv", "r") as inFile:
  for line in inFile:
     line = line.strip().split("\t")
     if len(line) < 3:
           print("ERROR", line)
           continue
     RoBERTa_alternatives[(line[0].strip(), line[1].strip())].append(line[2])



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
   original = alternative[0].strip()
   print(original)
   questionMarks = [int(x) for x in alternative[1].split(" ")]

   tokenizedBare = alternative[2].strip()
   tokenized = alternative[2].strip().split(" ")


   tokenized1 = tokenized[:questionMarks[0]]
   tokenized2 = tokenized[questionMarks[0]:]

   tokenizeds = [tokenized1, tokenized2]
   for i in range(2):
       tokenizeds[i] = ("".join(tokenizeds[i])).replace("▁", " ").replace("</s>", "").strip()
   tokenizedPairResult = tuple(tokenizeds) 
   original = tokenizedPairResult[0] + "@ " + tokenizedPairResult[1]

   print(original)
#   assert original in itemsPredictions
#   entry = itemsPredictions[original]
#   predictionForOriginal = torch.FloatTensor([float(x) for x in entry[2].split(" ")]).exp()
#   assert predictionForOriginal <= 1, entry
   #print(predictionForOriginal)
   #quit()

   hasConsideredSubsets = set()

   for variant in alternative[3:]:
      #print(variant)
      if len(variant) < 5:
         continue
      try:
         subset, sentence= variant.strip().split("\t")
      except ValueError:
        continue
      subset = subset.strip()
      sentence = sentence.split()
   #   print("SENTENCE AS FOUND", sentence)
      assert (subset,tokenizedBare) in RoBERTa_alternatives, (subset,tokenizedBare)

      if subset in hasConsideredSubsets:
        continue
      hasConsideredSubsets.add(subset)
      for sentence in RoBERTa_alternatives[(subset,tokenizedBare)]:
          sentence = sentence.strip()
#          print(sentence)
          variants_set.add(sentence)
          if subset not in variants_dict:
             variants_dict[subset] = []
          variants_dict[subset].append(sentence)
   print(len(variants_set), "variants")
   valuesPerVariant = {}
   for variant in variants_set:
   #  print(variant)
     try:
       assert alternatives_predictions_binary[variant] in [0, 1], alternatives_predictions_binary[variant]
       valuesPerVariant[variant] = alternatives_predictions_float[variant] 
     except ValueError:
        print("VALUE ERROR", variant)
        valuesPerVariant[variant] = 0
     except AttributeError:
        print("VALUE ERROR", variant)
        valuesPerVariant[variant] = 0

   varianceBySubset = {}
   for subset in variants_dict:
       values = torch.FloatTensor([ valuesPerVariant[x] for x in variants_dict[subset]]).exp()
       varianceBySubset[subset] = 4*float((values.mean(dim=0) - values).pow(2).mean(dim=0).max())
       assert varianceBySubset[subset] <= 1


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

   sensitivity = getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities)
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
variance = sum([x**2 for x in sensitivities]) / len(sensitivities) - (sum(sensitivities)/len(sensitivities))**2
import math
print("Standard error", variance/math.sqrt(len(sensitivities)))

sensitivityHistogram = torch.FloatTensor(sensitivityHistogram)
print(sensitivityHistogram/sensitivityHistogram.sum())


