import math
import sys
import torch
task = sys.argv[1]

assert task == "SST-2"

def mean(values):
   return sum(values)/len(values)

sensitivityHistogram = [0 for _ in range(40)]


def variance(values):
   values = values.exp()
   values = 2*values-1 # make probabilities rescale to [-1, 1]
   return float(((values-values.mean(dim=0)).pow(2).mean(dim=0)).sum())

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

averageLabel = [0,0,0]

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

for alternative in alternatives:
   if len(alternative) < 5:
      continue
   variants_set = set()
   variants_dict = {}
   
   alternative = alternative.split("\n")
   original = alternative[0]
   print(original)
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
   perSubsetSensitivities = [varianceBySubset[x] - 1e-5*len([y for y in x if y == "1"]) for x in subsetsEnumeration]

   sensitivity, assignment = getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities)
   if str(sensitivity) == "nan":
      continue
   print("OVERALL SENSITIVITY ON THIS DATAPOINT", sensitivity)
   print(tokenized)
   if sensitivity < 2 and False:
      continue
   subsetsBySensitivity = sorted(range(len(perSubsetSensitivities)), key=lambda x:perSubsetSensitivities[x], reverse=True)
   for i in subsetsBySensitivity:
      assigned = assignment[i].item()
      if assigned > 1e-2 and perSubsetSensitivities[i] > 0.0:
#         print(len(subsetsEnumeration[j]), len(tokenized))
         tokenized2 = ("".join([tokenized[j] if subsetsEnumeration[i][j] == "0" else "####" for j in range(len(tokenized))])).replace("▁", " ")
         print(tokenized2)
         print(subsetsEnumeration[i], assigned, perSubsetSensitivities[i])
         sentsWithValues = sorted([ (x, valuesPerVariant[x]) for x in variants_dict[subsetsEnumeration[i]]], key=lambda x:x[1])
         for y in sentsWithValues:
            print(y)
        

         sentences = [tokenized2]
         result = [[]]
         for s in range(1):
           sentence = sentences[s]
           while "####" in sentence:
              q = sentence.index("####")
              left, sentence = sentence[:q].strip(), sentence[q+4:].strip()
              if q == 0:
                 if len(result[s]) == 0:
                     result[s].append("####")
                 else:
                     result[s][-1] += "####"
              else:
                result[s].append(left)
                result[s].append("####")
           if len(sentence) > 0:
              result[s].append(sentence)
#         print({"premise" : result[0], "hypothesis" : result[1], "subset" : subsetsEnumeration[i], "original" : original}, ",", file=outFile)
   sensitivities.append(sensitivity)
   print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))
print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))
