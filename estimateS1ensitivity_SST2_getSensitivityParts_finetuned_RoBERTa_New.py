import random
rng = random.Random(5)
import math
import sys
import torch
task = sys.argv[1]

#from nltk.tokenize.treebank import TreebankWordDetokenizer
#detokenizer = TreebankWordDetokenizer()


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
   return -res.fun, res.x

from random import shuffle

alternatives_predictions_binary = {}
alternatives_predictions_float = {}

averageLabel = [0,0,0]


with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_datapoints_predictions_finetuned.tsv", "r") as inFile:
   itemsPredictions = dict([(x[0], x) for x in [x.split("\t") for x in inFile.read().strip().split("\n")]])

with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_alternatives_predictions_finetuned_RoBERTa.tsv", "r") as inFile:
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

alternatives = []
for group in [""]: #["c", "d", "e"]:
 with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_alternatives_c_sentBreak_new_finetuned_large.tsv", "r") as inFile:
  alternatives += inFile.read().strip().split("#####\n")
  print(len(alternatives))



from collections import defaultdict

RoBERTa_alternatives = defaultdict(list)
for group in [""]: # , "_d", "_e"
 with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/SST-2/dev_alternatives_RoBERTa_finetuned{group}.tsv", "r") as inFile:
  for line in inFile:
     line = line.strip().split("\t")
     if len(line) < 3:
           print("ERROR", line)
           continue
     RoBERTa_alternatives[(line[0].strip(), line[1].strip())].append(line[2])



sensitivities = []


processed = set()

with open(f"../block-certificates/items/witnesses_{__file__}", "w") as outFile_Witnesses:
 with open(f"/u/scr/mhahn/sensitivity/sensitivities/s1ensitivities_{__file__}", "w") as outFile:
  print("Original", "\t", "BinaryS1ensitivity", file=outFile)
  for alternative in alternatives:
   if len(alternative) < 5:
      continue
   variants_set = set()
   variants_dict = {}
   
   alternative = alternative.split("\n")
   original = alternative[0].strip()
   print("#######", file=outFile_Witnesses)
   print(original, file=outFile_Witnesses)
 #  print(original+"#")
   assert original in itemsPredictions
   tokenizedBare = alternative[1]
   tokenized = alternative[1].split(" ")
   valuesPerVariant = {}
   hasConsideredSubsets = set()

   #if tokenizedBare not in RoBERTa_alternatives_set:
   #   print("No predictions for this datapoint!", tokenizedBare)
   #   continue
   for variant in alternative[3:]:
      #print(variant)
      if len(variant) < 5:
         continue
      try:
         subset, sentence= variant.strip().split("\t")
      except ValueError:
        continue
      subset = subset.strip()
      if (subset,tokenizedBare) not in RoBERTa_alternatives:
          print("ERROR. If this happens more than a couple of times, then this is a problem", (subset,tokenizedBare))
          continue
      if ((subset, tokenizedBare) not in RoBERTa_alternatives):
         print("WEIRD", (subset, tokenizedBare))
      if (subset, tokenizedBare) in processed:
        continue
      if subset in hasConsideredSubsets:
        continue
      for sentence in RoBERTa_alternatives[(subset, tokenizedBare)]:
         #print(alternative)
         sentence = sentence.replace("<s>", "").replace("</s>", "").split("[SEP]")[0].strip()
         assert sentence in alternatives_predictions_float, sentence
#         if sentence not in alternatives_predictions_float:
 #           print("DID NOT FIND", sentence)
  #          #assert False
   #         continue
         valuesPerVariant[sentence] = alternatives_predictions_float[sentence]
         if subset not in variants_dict:
            variants_dict[subset] = []
         variants_dict[subset].append(sentence)
  # print((result))

   varianceBySubset = {}
   for subset in variants_dict:
       values = torch.stack([ valuesPerVariant[x] for x in variants_dict[subset]], dim=0).exp()
       varianceBySubset[subset] = 4*float((values.mean(dim=0) - values).pow(2).mean(dim=0).max())
       assert varianceBySubset[subset] <= 1
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
   print("OVERALL SENSITIVITY ON THIS DATAPOINT", sensitivity, file=outFile_Witnesses)
   print("OVERALL SENSITIVITY ON THIS DATAPOINT", sensitivity)
   print(tokenized)
#   if sensitivity < 2 and False:
 #     continue
   subsetsBySensitivity = sorted(range(len(perSubsetSensitivities)), key=lambda x:perSubsetSensitivities[x], reverse=True)
   print(subsetsBySensitivity)
   capturedSensitivity = 0
   for i in subsetsBySensitivity:
      assigned = assignment[i].item()
      if assigned > 1e-2 and perSubsetSensitivities[i] > 0.3:
         print("&&&&&&&&&&& SUBSET SENSITIVITY", "\t", assigned, "\t", float(perSubsetSensitivities[i]), file=outFile_Witnesses)
#         print(len(subsetsEnumeration[j]), len(tokenized))
         capturedSensitivity += perSubsetSensitivities[i]

         tokenized2 = [tokenized[j] if subsetsEnumeration[i][j] == "0" else "####" for j in range(len(tokenized))]
         
         tokenized2_1 = ("".join(tokenized2)).replace("▁", " ")
         print(tokenized2_1)



         sentences = [tokenized2_1]

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
         print("&&&&&&&&&@ SUBSETS", "\t", str("\t".join(sentences)), file=outFile_Witnesses)
         print("&&&&&&&&&% SUBSETS", "\t", str(result), file=outFile_Witnesses)


  #       print(subsetsEnumeration[i], assigned, perSubsetSensitivities[i])
         sentsWithValues = sorted([ (x, valuesPerVariant[x]) for x in variants_dict[subsetsEnumeration[i]]], key=lambda x:x[1])
         for sentence, prediction in sentsWithValues:
            sentence = sentence.split("[SEP]")[:1]
            #for i in range(1):
              #print(sentence)
              #quit()
              #sentence[i] = sentence[i].replace("[CLS]", "").replace("[SEP]", "").strip().replace(" ' s ", " 's ").replace(" ' ll ", " 'll ").replace(" ' d ", " 'd ").replace("n ' t ", "n't ").replace(" ' ve ", " 've ").replace(" @ - @ ", "-").replace("( ", "(").replace("U . S . ", "U.S. ")
              #sentence[i] = detokenizer.detokenize(sentence[i].split(" "))
               
            print("\t".join(sentence), "\t", prediction, file=outFile_Witnesses)
        

   sensitivities.append(sensitivity)
   print("Captured", capturedSensitivity, "out of", sensitivity)
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

print("Number of datapoints", len(sensitivities))
