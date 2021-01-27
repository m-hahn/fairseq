import math
import sys
import torch
task = sys.argv[1]
from nltk.tokenize.treebank import TreebankWordDetokenizer                                                                                                                                                  
detokenizer = TreebankWordDetokenizer()      
assert task == "SyntaxGym_260"

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
import random
from random import shuffle

alternatives_predictions_binary = {}
alternatives_predictions_float = {}

averageLabel = [0,0,0]

originalSentences = open(f"/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_260_alternatives_PMLM_1billion_raw_forGPT2.tsv", "r").read().strip().split("\n")
#originalSentences += open(f"/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_260_datapoints_raw.tsv", "r").read().strip().split("\n")

predictions = {}
sentencesInOrder = []
for f in ["alternatives"]:
 with open(f"/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_260_{f}_PMLM_1billion_raw_forGPT2_SURPRISALS.tsv", "r") as inFile:
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
    if random.random() < 0.001 or "amused" in sent and "next" in sent:
      print(len(sentencesInOrder), sent, originalSentences[len(sentencesInOrder)])
#    sent = sent.replace(" .", ".").replace(" ,", ",").replace(" :", ":").replace(" )", ")").replace("-- ", "--").replace(" '", "'").replace(" ;", ";").replace("` ", "`").replace(" n't", "n't")
#    print(len(sentencesInOrder), sent, originalSentences[len(sentencesInOrder)])
 #   if sent != originalSentences[len(sentencesInOrder)] and "<unk>" not in sent and "[" not in sent and '"' not in sent and "--" not in sent and "!" not in sent and ".." not in sent:
  #     print(len(sentencesInOrder), sent, originalSentences[len(sentencesInOrder)])
   #    assert False      
    if len(sentencesInOrder) == len(originalSentences):
       print(sent)
    origSent = originalSentences[len(sentencesInOrder)].replace("himself", "REFLEXIVE").replace("themselves", "REFLEXIVE")
    predictions[(origSent, reflexive)] = float(sentence[-3][3])
    sentencesInOrder.append(sent)
#print(predictions)
#quit()

from collections import defaultdict
alternativesPerVariant = defaultdict(list)
with open("/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_260_alternatives_PMLM_1billion_raw.tsv", "r") as inFile:
  for line in inFile:
     subset, original, sentence = line.strip().split("\t")
     sentence = sentence.replace('[CLS]', "").split("[SEP]")[0].replace("himself", "REFLEXIVE").replace("themselves", "REFLEXIVE").strip()
     original = original.replace("himself", "REFLEXIVE").replace("themselves", "REFLEXIVE").replace("</s>", "").strip()
     sentence = detokenizer.detokenize(sentence.split(" "))
     if "amused" in sentence:
       print(sentence)
     alternativesPerVariant[(subset.strip(), original)].append(sentence)

 #    if "▁next ▁to ▁the ▁senators ▁h" in original:
#       print((subset, original))
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
   if "amused" in sent:
     print((sent,))
#print(alternatives_predictions_float)

#assert 'He sat next to them , amused REFLEXIVE .' in alternatives_predictions_float


#quit()
print("Average Label", 2*averageLabel[1]/averageLabel[0]-1)
print("Label Variance", 4*(averageLabel[2]/averageLabel[0] - (averageLabel[1]/averageLabel[0])**2))

print(list(alternatives_predictions_float.items())[:10])



with open(f"/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_260_alternatives.tsv", "r") as inFile:
  alternatives = inFile.read().strip().split("#####\n")
  print(len(alternatives))

sensitivities = []

from collections import defaultdict
variants_set = defaultdict(set)
variants_dict = defaultdict(dict)
valuesPerVariant = {}



print(str(alternatives_predictions_float)[:500])

for alternative in alternatives:
   if len(alternative) < 5:
      continue
   alternative = alternative.split("\n")
   original = alternative[0]
   original = original.replace("himself", "REFLEXIVE").replace("themselves", "REFLEXIVE")
   tokenized = alternative[2].replace(" ", "").replace("▁", " ").replace("</s>", "").replace("himself", "REFLEXIVE").replace("themselves", "REFLEXIVE").strip()
   tokenizedOriginal = alternative[2].replace("</s>", "").replace("himself", "REFLEXIVE").replace("themselves", "REFLEXIVE").strip()

   print(tokenized)
   predictionOriginal = alternatives_predictions_float[tokenized]

   consideredSubsets = set()

   for variant in alternative[3:]:
      #print(variant)
      if len(variant) < 5:
         continue

      subset, sentence= variant.strip().split("\t")
      subset = subset.strip()
      if subset in consideredSubsets:
          continue
      consideredSubsets.add(subset)
     
  #    print(str(alternativesPerVariant)[:500])
 #     print(subset, tokenizedOriginal) 
      assert (subset, tokenizedOriginal) in alternativesPerVariant, (subset, tokenizedOriginal)
      alternativesForSubset = alternativesPerVariant[(subset, tokenizedOriginal)]
#      print(alternativesForSubset)

      for sentence in alternativesForSubset:
         variants_set[original].add(sentence)
         if subset not in variants_dict[original]:
            variants_dict[original][subset] = []
         variants_dict[original][subset].append(sentence)
  # print((result))
   print(len(variants_set[original]), "variants")
#   assert 'He sat next to them , amused REFLEXIVE .' in alternatives_predictions_float
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

with open(f"/u/scr/mhahn/sensitivity/sensitivities/s1ensitivities_{__file__}", "w") as outFile:
 print("Original", "\t", "BinaryS1ensitivity", file=outFile)
 for alternative in alternatives:
   if len(alternative) < 5:
      continue
   alternative = alternative.split("\n")
   original = alternative[0]
   original = original.replace("himself", "REFLEXIVE").replace("themselves", "REFLEXIVE")
   print(original)
   tokenized = alternative[2].split(" ")
   tokenized = alternative[2].replace(" ", "").replace("▁", " ").replace("</s>", "").replace("himself", "REFLEXIVE").replace("themselves", "REFLEXIVE").strip()

   print(tokenized)
   predictionOriginal = alternatives_predictions_float[tokenized]


   varianceBySubset = {}
   for subset in variants_dict[original]:
       values = torch.FloatTensor([ valuesPerVariant[x] for x in variants_dict[original][subset]])
       assert float(values.abs().max()) <= 1
       assert abs(predictionOriginal) <= 1
       varianceBySubset[subset] = 4*float((values.mean(dim=0)-values).pow(2).mean(dim=0).max())




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

   sensitivity, _ = getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities)
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

