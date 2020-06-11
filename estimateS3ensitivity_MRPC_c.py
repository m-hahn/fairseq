import math
import sys
task = sys.argv[1]

assert task == "MRPC"

def mean(values):
   return sum(values)/len(values)

sensitivityHistogram = [0 for _ in range(40)]

import torch
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



with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/MRPC/dev_datapoints_predictions_fairseq.tsv", "r") as inFile:
   itemsPredictions = dict([(x[0]+"@ "+x[1], x) for x in [x.split("\t") for x in inFile.read().strip().split("\n")]])

with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/MRPC/dev_alternatives_c_predictions_fairseq.tsv", "r") as inFile:
  for line in inFile:
     if len(line) < 5:
       continue
     line = line.strip().split("\t")
     if len(line) == 2:
       line.append("0.0")
     sentence1, sentence2, cont, binary = line
     sentence = sentence1+" "+sentence2
     cont = float(cont)
     assert cont < 0
     alternatives_predictions_binary[sentence.strip()] = binary.strip()
     alternatives_predictions_float[sentence.strip()] = float(cont)
  print(len(alternatives_predictions_binary))

print(list(alternatives_predictions_binary.items())[:10])

with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/MRPC/dev_alternatives_c.tsv", "r", encoding='utf-8') as inFile:
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
   original = alternative[0].replace("</s>", "").strip()
   print(original)
   questionMarks = [int(x) for x in alternative[1].split(" ")]

   tokenized = alternative[2].replace("</s>", "").strip().split(" ")

   print(original)
   assert original in itemsPredictions
   entry = itemsPredictions[original]
   predictionForOriginal = math.exp(float(entry[2])) #torch.FloatTensor([float(x) for x in entry[3].split(" ")])
 


   for variant in alternative[3:]:
      #print(variant)
      if len(variant) < 5:
         continue

      subset, sentence= variant.strip().split("\t")

      sentence = sentence.strip().split(" ")
      sentence1 = sentence[:questionMarks[0]]
      sentence2 = sentence[questionMarks[0]:]

      sentences = [sentence1, sentence2]
      for i in range(2):
          sentences[i] = ("".join(sentences[i])).replace("▁", " ").replace("</s>", "").strip()
      sentencePairResult = tuple(sentences) 
      sentence = sentencePairResult[0] + " " + sentencePairResult[1]
      if sentence not in alternatives_predictions_binary:
         print("DID NOT FIND", sentence)
         assert False
         continue
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
       assert alternatives_predictions_binary[variant] in ["0", "1"], alternatives_predictions_binary[variant]
#       valuesPerVariant[variant] = 1 if alternatives_predictions_binary[variant] == "1" else -1
       valuesPerVariant[variant] = float(alternatives_predictions_float[variant] )
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
       values = torch.FloatTensor([ valuesPerVariant[x] for x in variants_dict[subset]]).exp()
       #print(subset, mean(values), variance(values))
 #      print(predictionForOriginal)
  #     print(values, values.size())
      
#       quit()
       varianceBySubset[subset] = 4*float((values.mean(dim=0)-predictionForOriginal).pow(2).max())
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

print("Examples", len(sensitivities))
print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))
print("Median block sensitivity of the model", sorted(sensitivities)[int(len(sensitivities)/2)])


sensitivityHistogram = torch.FloatTensor(sensitivityHistogram)
print(sensitivityHistogram/sensitivityHistogram.sum())


