import math
import sys
import torch
task = sys.argv[1]

assert task == "CB"

def mean(values):
   return sum(values)/len(values)

sensitivityHistogram = [0 for _ in range(40)]


def variance(values):
   values = values.exp()
   return float(((values-values.mean(dim=0)).pow(2).mean(dim=0)).sum())

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
with open(f"/u/scr/mhahn/PRETRAINED/SuperGLUE/CB/dev_alternatives_predictions_fairseq.tsv", "r", encoding='utf-8') as inFile:
  for line in inFile:
     if len(line) < 5:
       continue
     line = line.strip().split("\t")
     if len(line) == 2:
       line.append("0.0")
     sentence, hypothesis, predictions = line
     predictions = [float(x) for x in predictions.split(" ")]
     alternatives_predictions_float[sentence.strip()] = torch.FloatTensor(predictions)
  print(len(alternatives_predictions_float))

print(list(alternatives_predictions_float.items())[:10])

with open(f"/u/scr/mhahn/PRETRAINED/SuperGLUE/CB/dev_alternatives_c.tsv", "r", encoding='utf-8', errors='replace') as inFile:
  alternatives = inFile.read().strip().split("#####\n")
  print(len(alternatives))

sensitivities = []

with open(f"/u/scr/mhahn/sensitivity/sensitivities/sensitivities_{__file__}", "w") as outFile:
 print("Original", "\t", "BinarySensitivity", file=outFile)
 for alternative in alternatives:
   if len(alternative) < 5:
      continue
   variants_set = set()
   variants_dict = {}
   
   alternative = alternative.split("\n")
   original = alternative[0]
   print(original)
   questionMarks = [int(x) for x in alternative[1].split(" ")]

   tokenized = alternative[1].split(" ")
   for variant in alternative[3:]:
      #print(variant)
      if len(variant) < 5:
         continue

      subset, sentence= variant.strip().split("\t")
      sentence = sentence.strip().split(" ")
      sentence = "".join(sentence)
      sentence = sentence.replace("▁", " ").strip()
      #print("##"+sentence+"##")
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
#       valuesPerVariant[variant] = float(alternatives_predictions_float[variant] )
     #  if len(valuesPerVariant) % 100 == 0:
      #   print(valuesPerVariant[variant], valuesPerVariant[variant] == True, len(valuesPerVariant), len(variants_set), variant)
     except ValueError:
        print("VALUE ERROR", variant)
        valuesPerVariant[variant] = 0
     except AttributeError:
        print("VALUE ERROR", variant)
        valuesPerVariant[variant] = 0

   varianceBySubset = {}
   for subset in variants_dict:
       values = torch.stack([ valuesPerVariant[x] for x in variants_dict[subset]], dim=0)
       #print(subset, mean(values), variance(values))
       varianceBySubset[subset] = variance(values)
#   print(varianceBySubset)


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
   sensitivityHistogram[int(2*sensitivity)] += 1
   sensitivities.append(sensitivity)
   print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))
   print(original, "\t", sensitivity, file=outFile)

print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))
print("Median block sensitivity of the model", sorted(sensitivities)[int(len(sensitivities)/2)])


sensitivityHistogram = torch.FloatTensor(sensitivityHistogram)
print(sensitivityHistogram/sensitivityHistogram.sum())


