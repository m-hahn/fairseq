import math
import sys
import torch
task = sys.argv[1]

assert task == "SyntaxGym_248"

def mean(values):
   return sum(values)/len(values)

sensitivityHistogram = [0 for _ in range(40)]


def variance(values):
   values = 2*values-1 # make probabilities rescale to [-1, 1]
   return float(((values-values.mean(dim=0)).pow(2).mean(dim=0)).max())

from scipy.optimize import linprog


def getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities):
   #print(perSubsetSensitivities)
   c = [-x for x in perSubsetSensitivities]
   res = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds)
   # find the highly sensitive partition
   return -res.fun, res.x

import random
from random import shuffle

alternatives_predictions_binary = {}
alternatives_predictions_float = {}

averageLabel = [0,0,0]



originalSentences = open(f"/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_248_alternatives_raw.tsv", "r").read().strip().split("\n")

predictions = {}
sentencesInOrder = []
with open(f"/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_248_alternatives_raw_gpt2.tsv", "r") as inFile:
  header = next(inFile)
  assert header == "sentence_id\ttoken_id\ttoken\tsurprisal\n"
  data2 = inFile.read().split("\n")
  sentence_id = "1"
  data = [""]
  for x in data2:
     if "\t" not in x:
        #print(x)
        continue
     if x[:x.index("\t")] != sentence_id:
        data.append("")
     sentence_id = x[:x.index("\t")]
     data[-1] += x+"\n"
  for sentence in data:
    sentence = [x.split("\t") for x in sentence.split("\n")]
    #print(sentence)
    #quit()
    if len(sentence) < 3 or len(sentence[-3]) < 3:
      print("SHORT", sentence)
      continue
    assert sentence[-3][2] in ["Ġhimself", "Ġthemselves"], sentence[-3][2]
    reflexive = sentence[-3][2].replace("Ġ", "")
    #sentence[-3][2] = "REFLEXIVE"
    sent = " ".join([x[2] for x in sentence if len(x) > 2]).strip()
#    sent = sent.replace("himself", "REFLEXIVE")
 #   sent = sent.replace("themselves", "REFLEXIVE")
    if random.random() < 0.001:
      print(len(sentencesInOrder), sent, originalSentences[len(sentencesInOrder)])
    sent = sent.replace(" .", ".").replace(" ,", ",").replace(" :", ":").replace(" )", ")").replace("-- ", "--").replace(" '", "'").replace(" ;", ";").replace("` ", "`").replace(" n't", "n't")
#    print(len(sentencesInOrder), sent, originalSentences[len(sentencesInOrder)])
 #   if sent != originalSentences[len(sentencesInOrder)] and "<unk>" not in sent and "[" not in sent and '"' not in sent and "--" not in sent and "!" not in sent and ".." not in sent:
  #     print(len(sentencesInOrder), sent, originalSentences[len(sentencesInOrder)])
   #    assert False      
    if len(sentencesInOrder) == len(originalSentences):
       print(sent)
    origSent = originalSentences[len(sentencesInOrder)].replace("himself", "REFLEXIVE").replace("themselves", "REFLEXIVE")
    predictions[(origSent, reflexive)] = float(sentence[-3][3])
    sentencesInOrder.append(sent)
#quit()

#print(predictions)
#quit()

sentences = [x[0] for x in predictions]
for sent in sentences:
   himself = -predictions[(sent, "himself")]
   themselves = -predictions[(sent, "themselves")]
   himselfNorm = math.exp(himself)/(math.exp(himself)+math.exp(themselves))
   alternatives_predictions_float[sent] = himselfNorm
   averageLabel[0] += 1
   averageLabel[1] += himselfNorm
   averageLabel[2] += himselfNorm**2
#print(alternatives_predictions_float)
#quit()
print("Average Label", 2*averageLabel[1]/averageLabel[0]-1)
print("Label Variance", 4*(averageLabel[2]/averageLabel[0] - (averageLabel[1]/averageLabel[0])**2))

print(list(alternatives_predictions_float.items())[:10])

with open(f"/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_248_alternatives.tsv", "r") as inFile:
  alternatives = inFile.read().strip().split("#####\n")
  print(len(alternatives))

sensitivities = []

from collections import defaultdict
variants_set = defaultdict(set)
variants_dict = defaultdict(dict)
valuesPerVariant = {}


for alternative in alternatives:
   if len(alternative) < 5:
      continue
   alternative = alternative.split("\n")
   original = alternative[0]
   original = original.replace("himself", "REFLEXIVE").replace("themselves", "REFLEXIVE")
   print(original)
   tokenized = alternative[2].split(" ")
   for variant in alternative[3:]:
      #print(variant)
      if len(variant) < 5:
         continue

      subset, sentence= variant.strip().split("\t")
      sentence = "".join(sentence.strip().split(" "))
      sentence = sentence.replace("▁", " ").replace("</s>", "").replace("himself", "REFLEXIVE").replace("themselves", "REFLEXIVE")
      sentence = sentence.strip()
      # for GRNN, I manually removed quotation marks for lm-zoo tokenizer
      sentence = sentence.replace('"', "")
      if sentence not in alternatives_predictions_float:
         print("DID NOT FIND", sentence)
         assert False
         continue
      assert sentence in alternatives_predictions_float, sentence


      variants_set[original].add(sentence)
      if subset not in variants_dict[original]:
         variants_dict[original][subset] = []
      variants_dict[original][subset].append(sentence)
  # print((result))
   print(len(variants_set[original]), "variants")
   for variant in variants_set[original]:
   #  print(variant)
     try:
       valuesPerVariant[variant] = alternatives_predictions_float[variant]
     except ValueError:
        print("VALUE ERROR", variant)
        valuesPerVariant[variant] = 0
     except AttributeError:
        print("VALUE ERROR", variant)
        valuesPerVariant[variant] = 0

#   print(varianceBySubset)

#with open(f"/u/scr/mhahn/sensitivity/sensitivities/sensitivities_{__file__}", "w") as outFile:
#print("Original", "\t", "BinarySensitivity", file=outFile)
for alternative in alternatives:
   if len(alternative) < 5:
      continue
   alternative = alternative.split("\n")
   original = alternative[0]
   original = original.replace("himself", "REFLEXIVE").replace("themselves", "REFLEXIVE")
   print(original)
   tokenized = alternative[2].split(" ")

   varianceBySubset = {}
   for subset in variants_dict[original]:
       values = torch.FloatTensor([ valuesPerVariant[x] for x in variants_dict[original][subset]])
       #print(subset, mean(values), variance(values))
       varianceBySubset[subset] = variance(values)




   subsetsEnumeration = list(variants_dict[original])
   if len(subsetsEnumeration) == 0:
     continue 
   N = len(subsetsEnumeration[0])
   A = [[0 for subset in range(len(subsetsEnumeration))] for inp in range(N)]
   for inp in range(N):
       for subset, bitstr in enumerate(subsetsEnumeration):
#          print(bitstr, N)
          assert len(bitstr) == N
          if bitstr[inp] == "1":
              A[inp][subset] = 1
   
   
   b = [1 for _ in range(N)]
   x_bounds = [(0,1) for _ in range(len(subsetsEnumeration))]
   perSubsetSensitivities = [varianceBySubset[x] for x in subsetsEnumeration]

   sensitivity, assignment = getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities)
   print("OVERALL SENSITIVITY ON THIS DATAPOINT", sensitivity)
   sensitivityHistogram[int(2*sensitivity)] += 1
   print("===")
   for i in range(len(subsetsEnumeration)):
      assigned = assignment[i].item()
      if assigned > 1e-2 and perSubsetSensitivities[i] > 0.0:
#         print(len(subsetsEnumeration[j]), len(tokenized))
         tokenized2 = ("".join([tokenized[j] if subsetsEnumeration[i][j] == "0" else "####" for j in range(len(tokenized))])).replace("▁", " ")
         print(tokenized2)
         print(subsetsEnumeration[i], "Weight:", assigned, "Sensitivity:", perSubsetSensitivities[i])
         print(variants_dict[original][subsetsEnumeration[i]])
         sentences = tokenized2.split("@")
         result = [[], []]
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
              print(result[0])
#         print({"premise" : result[0], "hypothesis" : result[1], "subset" : subsetsEnumeration[i], "original" : original}, ",", file=outFile)
   sensitivities.append(sensitivity)
   print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))
#   print(original, "\t", sensitivity, file=outFile)

print("Average block sensitivity of the model", sum(sensitivities)/len(sensitivities))
print("Median block sensitivity of the model", sorted(sensitivities)[int(len(sensitivities)/2)])

import torch
sensitivityHistogram = torch.FloatTensor(sensitivityHistogram)
print(sensitivityHistogram/sensitivityHistogram.sum())


