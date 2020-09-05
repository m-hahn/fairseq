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
   return -res.fun, res.x

from random import shuffle

alternatives_predictions_binary = {}
alternatives_predictions_float = {}
predictions_all = []


#with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_datapoints_predictions_fairseq.tsv", "r") as inFile:
#   itemsPredictions = dict([(x[0]+"@ "+x[1], x) for x in [x.split("\t") for x in inFile.read().strip().split("\n")]])


with open(f"/u/scr/mhahn/PRETRAINED/WSC/val_alternatives_predictions_raw_RoBERTa.txt", "r") as inFile:
  for line in inFile:
     if len(line) < 5:
       continue
     line = line.strip().split("\t")
     if len(line) == 2:
       line.append("0.0")
     sentence, prediction_log, prediction_discrete = line
     alternatives_predictions_float[sentence.strip()] = torch.FloatTensor([float(prediction_log)])
     #averageLabel[0]+=1
     #averageLabel[1]+=math.exp(float(prediction_log))
     #averageLabel[2]+=(math.exp(float(prediction_log)))**2
  print(len(alternatives_predictions_float))

#print("Average Label", 2*averageLabel[1]/averageLabel[0]-1)
#print("Label Variance", 4*(averageLabel[2]/averageLabel[0] - (averageLabel[1]/averageLabel[0])**2))




predictions_all = torch.FloatTensor(predictions_all)
variance_predictions = predictions_all.pow(2).mean(dim=0) - predictions_all.mean(dim=0).pow(2)
print(predictions_all)
print(predictions_all.mean(dim=0))
print(variance_predictions)
#quit()




with open(f"/u/scr/mhahn/PRETRAINED/WSC/val_alternatives_c.txt", "r", encoding='utf-8') as inFile:
  alternatives = inFile.read().strip().split("#####\n")
  print(len(alternatives))



from collections import defaultdict

RoBERTa_alternatives = defaultdict(list)
for group in [""]: # , "_d", "_e"
 with open(f"/u/scr/mhahn/PRETRAINED/WSC/val_alternatives_RoBERTa_raw{group}.tsv", "r") as inFile:
  for line in inFile:
     line = line.strip().split("\t")
     if len(line) < 3:
           print("ERROR", line)
           continue
     line[1] = line[1].replace("</s>", "")
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
 #  print(original+"#")
   #entry = itemsPredictions[original]
#   predictionForOriginal = float(entry[2])
 #  assert predictionForOriginal <= 0
  # print(entry)
   boundaries = alternative[1].strip()
   tokenized2 = alternative[2].strip()
   tokenized = alternative[2].strip().split(" ")
   tokenized2 = tokenized[::]
   valuesPerVariant = {}
   boundaries = [int(x) for x in boundaries.split(" ")]
   if boundaries[0] > boundaries[2]:
    tokenized2.insert(boundaries[1], "]") # Semantically, that's the wrong way, but that's how the RoBERTa alternatives file does it
    tokenized2.insert(boundaries[0], "[")
    tokenized2.insert(boundaries[3], "_")
    tokenized2.insert(boundaries[2], "_")
   else:
    tokenized2.insert(boundaries[3], "]")
    tokenized2.insert(boundaries[2], "[")
    tokenized2.insert(boundaries[1], "_")
    tokenized2.insert(boundaries[0], "_")
   tokenized2 = " ".join(tokenized2)
   for variant in alternative[3:]:
      #print(variant)
      if len(variant) < 5:
         continue
      try:
         subset, sentence= variant.strip().split("\t")
      except ValueError:
         continue
      subset = subset.strip()
      if (subset, tokenized2) in processed:
        continue
      processed.add((subset, tokenized2))
      if ((subset, tokenized2) not in RoBERTa_alternatives):
         print("WEIRD", (subset, tokenized2))
         print("FROM VARIANT", [(subset, tokenized2)])
         print("ALTERNATIVES", list(RoBERTa_alternatives)[:1])
         #assert False
      for alternative in RoBERTa_alternatives[(subset, tokenized2)]:
         #print(alternative)
         alternative = alternative.replace("<s>", "").replace("</s>", "").strip()
         if alternative not in alternatives_predictions_float:
            print("DID NOT FIND", alternative)
            #assert False
            continue
         valuesPerVariant[alternative] = alternatives_predictions_float[alternative]
         if subset not in variants_dict:
            variants_dict[subset] = []
         variants_dict[subset].append(alternative)
  # print((result))

   varianceBySubset = {}
   for subset in variants_dict:
       values = torch.stack([ valuesPerVariant[x] for x in variants_dict[subset]], dim=0)
       varianceBySubset[subset] = float((values.mean(dim=0) - values).pow(2).mean(dim=0).max())
       assert varianceBySubset[subset] <= 1, values
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


