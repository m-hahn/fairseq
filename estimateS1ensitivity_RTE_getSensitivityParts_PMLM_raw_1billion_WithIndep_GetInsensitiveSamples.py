import math
import sys
import torch
task = sys.argv[1]

from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()


assert task == "RTE"

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

with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_datapoints_predictions_fairseq.tsv", "r") as inFile:
   itemsPredictions = dict([(x[0]+"@ "+x[1], x) for x in [x.split("\t") for x in inFile.read().strip().split("\n")]])



for group in ["", "_Independent"]:
 with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_alternatives_predictions_PMLM_1billion_raw{group}.tsv", "r", encoding='utf-8') as inFile:
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
#quit()

predictions_all = torch.FloatTensor(predictions_all)
variance_predictions = predictions_all.pow(2).mean(dim=0) - predictions_all.mean(dim=0).pow(2)
print(predictions_all)
print(predictions_all.mean(dim=0))
print(variance_predictions)

alternatives = []
#quit()

for group in ["", "_OnlySubsetsNoAlternatives"]:
 with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_alternatives_c{group}.tsv", "r", encoding='utf-8') as inFile:
  alternatives += inFile.read().strip().split("#####\n")
  print(len(alternatives))



from collections import defaultdict

RoBERTa_alternatives_set = set()
RoBERTa_alternatives = defaultdict(list)
for group in ["", "_Independent"]: # , "_d", "_e"
 with open(f"/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_alternatives_PMLM_1billion_raw{group}.tsv", "r") as inFile:
  for line in inFile:
     line = line.strip().split("\t")
     if len(line) < 3:
           print("ERROR", line)
           continue
     RoBERTa_alternatives[(line[0].strip(), line[1].strip())].append(line[2])
     RoBERTa_alternatives_set.add(line[1].strip())


sensitivities = []


processed = set()

with open(f"../block-certificates/items/insensitive_witnesses_{__file__}", "w") as outFile_Witnesses:
# with open(f"/u/scr/mhahn/sensitivity/sensitivities/s1ensitivities_{__file__}", "w") as outFile:
#  print("Original", "\t", "BinaryS1ensitivity", file=outFile)
  for alternative in alternatives:
   if len(alternative) < 5:
      continue
   variants_set = set()
   variants_dict = {}
   
   alternative = alternative.split("\n")
   original = alternative[0].strip()
   print("#######", file=outFile_Witnesses)
   print(original, file=outFile_Witnesses)
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
   assert original in itemsPredictions
   entry = itemsPredictions[original]
   predictionForOriginal = torch.FloatTensor([float(x) for x in entry[2].split(" ")]).exp()
   assert predictionForOriginal <= 1, entry
   #print(predictionForOriginal)
   #quit()

   hasConsideredSubsets = set()

   if tokenizedBare not in RoBERTa_alternatives_set:
      print("No predictions for this datapoint!", tokenizedBare)
      continue
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
          assert sentence in alternatives_predictions_binary, sentence
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
       assert predictionForOriginal >= 0
#       print(values, predictionForOriginal)
       varianceBySubset[subset] = 4*float((values.mean(dim=0) - predictionForOriginal).pow(2).max())
       assert varianceBySubset[subset] <= 4, varianceBySubset[subset]
 #      print(varianceBySubset[subset])
  #     quit()

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
   perSubsetSensitivities = [(1 - varianceBySubset[x]) + 1e-5*len([y for y in x if y == "1"]) for x in subsetsEnumeration]

   sensitivity, assignment = getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities)
#   print("OVERALL SENSITIVITY ON THIS DATAPOINT", sensitivity, file=outFile_Witnesses)
 #  print("OVERALL SENSITIVITY ON THIS DATAPOINT", sensitivity)
   print(tokenized)
#   if sensitivity < 2 and False:
 #     continue
   subsetsBySensitivity = sorted(range(len(perSubsetSensitivities)), key=lambda x:perSubsetSensitivities[x], reverse=True)
   print(subsetsBySensitivity)
   capturedSensitivity = 0
   def disjoint(x,y):
      for i in range(len(x)):
        if x[i] == "1" and x[i] == y[i]:
            return False
      return True
   def maskSum(x, y):
      return "".join(["0" if x[i] == y[i] and x[i] == "0" else "1" for i in range(len(x))])

   currentMask = "".join(["0" for _ in subsetsEnumeration[0]])
   selectedSubsets = []
   for _ in range(5):
       insensitiveSubsets = sorted([i for i in subsetsBySensitivity if disjoint(currentMask, subsetsEnumeration[i]) and varianceBySubset[subsetsEnumeration[i]] < 1e-3], key=lambda i : -len([y for y in subsetsEnumeration[i] if y == "1"]))
       print(insensitiveSubsets)
       if len(insensitiveSubsets) == 0:
         break
       newSubset = subsetsEnumeration[insensitiveSubsets[0]]
       selectedSubsets.append(insensitiveSubsets[0])
       print(newSubset)
       currentMask = maskSum(newSubset, currentMask)
       print("mask", currentMask)
   for i in selectedSubsets:
#         print("&&&&&&&&&&& SUBSET SENSITIVITY", "\t", varianceBySubset[subsetsEnumeration[i]], file=outFile_Witnesses)

         tokenized2 = [tokenized[j] if subsetsEnumeration[i][j] == "0" else "####" for j in range(len(tokenized))]
         
         tokenized2_1 = ("".join(tokenized2[:questionMarks[0]])).replace("▁", " ")
         tokenized2_2 = ("".join(tokenized2[questionMarks[0]:])).replace("▁", " ")
         print(tokenized2_1, tokenized2_2)



         sentences = [tokenized2_1, tokenized2_2]

         result = [[], []]
         for s in range(2):
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



