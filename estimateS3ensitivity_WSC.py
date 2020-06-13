import math
import sys
import torch
task = sys.argv[1]

assert task == "WSC"

def mean(values):
   return sum(values)/len(values)

sensitivityHistogram = [0 for _ in range(40)]

def variance(values):
   return mean([x**2 for x in values]) - mean(values)**2

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


#with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_datapoints_predictions_fairseq.tsv", "r") as inFile:
#   itemsPredictions = dict([(x[0]+"@ "+x[1], x) for x in [x.split("\t") for x in inFile.read().strip().split("\n")]])

with open(f"/u/scr/mhahn/PRETRAINED/WSC/val_alternatives_predictions_fairseq.txt", "r", encoding='utf-8') as inFile:
  for line in inFile:
     if len(line) < 5:
       continue
     line = line.strip().split("\t")
     #print(line)
     sentence, binary = line
     binary = binary.strip()
     if binary in ["True", "False"]:
        binary = {"True" : 1, "False" : -1}[binary]
     binary = float(binary)
     assert binary in [-1, 1]
     sentence = sentence.replace("_ ,", "_,").replace(". . .", "...")
     sentence = sentence.replace("_", "").replace("[", "").replace("]", "").replace(" ", "")
#     print(sentence)
     alternatives_predictions_binary[sentence.strip()] = int(binary)
     alternatives_predictions_float[sentence.strip()] = None
     predictions_all.append(binary)
  print(len(alternatives_predictions_binary))

predictions_all = torch.FloatTensor(predictions_all)
variance_predictions = predictions_all.pow(2).mean(dim=0) - predictions_all.mean(dim=0).pow(2)
print(predictions_all)
print(predictions_all.mean(dim=0))
print(variance_predictions)
#quit()

with open(f"/u/scr/mhahn/PRETRAINED/WSC/val_alternatives_c.txt", "r", encoding='utf-8') as inFile:
  alternatives = inFile.read().strip().split("#####\n")
  print(len(alternatives))

sensitivities = []

with open(f"/u/scr/mhahn/sensitivity/sensitivities/s3ensitivities_{__file__}", "w") as outFile:
 print("Original", "\t", "BinaryS3ensitivity", file=outFile)
 for alternative in alternatives:
   if len(alternative) < 5:
      continue
   variants_set = set()
   variants_dict = {}
   
   alternative = alternative.split("\n")
   original = alternative[0].strip()
#   print(original)
   questionMarks = [int(x) for x in alternative[1].split(" ")]

   tokenized = alternative[2].strip()

 #  print(tokenized.replace(" ", "").replace("▁", " ")+"###")
#   print(tokenized.replace(" ", "").replace("▁", " ").replace(" ,", ",")+"###")
   tokenized = tokenized.replace(" ", "").replace("▁", " ").replace(" ,", ",").replace("_", "").replace("[", "").replace("]", "").strip().replace(" ", "")
 



   print("TOKENIZED", tokenized+"###")
   #print(list(alternatives_predictions_binary.items())[:10])
   assert tokenized in alternatives_predictions_binary
   entry = alternatives_predictions_binary[tokenized]
   predictionForOriginal = entry
   assert predictionForOriginal <= 1, entry


   for variant in alternative[3:]:
      #print(variant)
      if len(variant) < 5:
         continue

      subset, sentence= variant.strip().split("\t")
      sentence = sentence.replace("_", "").replace("[", "").replace("]", "").replace(" ", "").replace("▁", " ").replace(" ,", ",").replace(" .", ".").replace(" ", "").strip()

      if sentence not in alternatives_predictions_binary:
         print("DID NOT FIND", sentence)
#         assert False
         continue
      else:
#         print("FOUND", sentence)
         pass
#         print("FOUND", sentence)
      assert sentence in alternatives_predictions_binary, sentence


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
       assert alternatives_predictions_binary[variant] in [-1, 1], alternatives_predictions_binary[variant]
       valuesPerVariant[variant] = alternatives_predictions_binary[variant] 
     except ValueError:
        print("VALUE ERROR", variant)
        valuesPerVariant[variant] = 0
     except AttributeError:
        print("VALUE ERROR", variant)
        valuesPerVariant[variant] = 0

   varianceBySubset = {}
   for subset in variants_dict:
       values = torch.FloatTensor([ valuesPerVariant[x] for x in variants_dict[subset]])
       varianceBySubset[subset] = float((values.mean(dim=0)-predictionForOriginal).pow(2).max())
       assert varianceBySubset[subset] <= 4, varianceBySubset[subset]


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


sensitivityHistogram = torch.FloatTensor(sensitivityHistogram)
print(sensitivityHistogram/sensitivityHistogram.sum())


