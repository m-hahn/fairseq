import math
import sys
import torch
task = sys.argv[1]

assert task == "SyntaxGym_259"

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
   return -res.fun
import random
from random import shuffle

alternatives_predictions_binary = {}
alternatives_predictions_float = {}

averageLabel = [0,0,0]



with open("/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_259_alternatives.tsv", "r") as inFile:                                                                                                               
    alternatives = inFile.read().strip().split("#####\n")                                                                                                                                                   
                                                                                                                                                                                                            
versions = [[]]                                                                                                                                                                                             
                                                                                                                                                                                                            
count = 0                                                                                                                                                                                                   
for alt in alternatives:                                                                                                                                                                                    
  if len(alt) < 5:                                                                                                                                                                                          
    print("SKIPPED")                                                                                                                                                                                        
    continue                                                                                                                                                                                                
  alt = alt.strip().split("\n")[0]                                                                                                                                                                          
  remainder = alt[alt.index("@"):].replace("</s>", "").replace("@", "").strip()                                                                                                                             
  print(remainder)                                                                                                                                                                                          
  versions[int(count/2)].append(remainder)                                                                                                                                                                  
  if count % 2== 1:                                                                                                                                                                                         
     versions.append([])                                                                                                                                                                                    
  count += 1                                                                                                                                                                                                
print(list(set([tuple([y.split(" ")[0] for y in x]) for x in versions])))                                                                                                                                   
conversion = (dict(list(set([tuple([y for y in x]) for x in versions if len(x) >1])) + list(set([tuple([y for y in x][::-1]) for x in versions if len(x) >1]))) )

originalSentences = open(f"/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_259_alternatives_raw.tsv", "r").read().strip().split("\n")

predictions = {}
sentencesInOrder = []
with open(f"/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_259_alternatives_raw_gpt2.tsv", "r") as inFile:
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
    #print(sentence)
    #print(conversion)
    assert sentence[-2][2] == "."
    search = "."
    for j in range(len(sentence)-3, -1, -1):
       search = " "+sentence[j][2].replace("Ġ", "") + search
       if search.strip() in conversion:
           break
    search = search.strip()
    sent = " ".join([x[2] for x in sentence if len(x) > 2]).strip()
    if random.random() < 0.001:
      print(len(sentencesInOrder), sent, originalSentences[len(sentencesInOrder)])
    sent = sent.replace(" .", ".").replace(" ,", ",").replace(" :", ":").replace(" )", ")").replace("-- ", "--").replace(" '", "'").replace(" ;", ";").replace("` ", "`").replace(" n't", "n't")
    if len(sentencesInOrder) == len(originalSentences):
       print(sent)
    origSent = originalSentences[len(sentencesInOrder)]
    origSent = origSent[:-len(search)] + "VERB"
 #   print(origSent)
#    quit()
    verb = search[:search.index(" ")]
    if verb.endswith("s"):
      number = "SINGULAR"
    else:
      number = "PLURAL"
    predictions[(origSent, number)] = float(sentence[-3][3])
    sentencesInOrder.append(sent)
#quit()

#for x in predictions.items():
#  print(x)
#quit()
#print(sentencesInOrder)
#quit()

sentences = [x[0] for x in predictions]
for sent in sentences:
   if (sent, "SINGULAR") not in predictions or (sent, "PLURAL") not in predictions:
       print("MISSING", sent)
       continue
   SINGULAR = -predictions[(sent, "SINGULAR")]
   PLURAL = -predictions[(sent, "PLURAL")]
   SINGULARNorm = math.exp(SINGULAR)/(math.exp(SINGULAR)+math.exp(PLURAL))
   alternatives_predictions_float[sent] = SINGULARNorm
   averageLabel[0] += 1
   averageLabel[1] += SINGULARNorm
   averageLabel[2] += SINGULARNorm**2
#print(alternatives_predictions_float)
#quit()
print("Average Label", 2*averageLabel[1]/averageLabel[0]-1)
print("Label Variance", 4*(averageLabel[2]/averageLabel[0] - (averageLabel[1]/averageLabel[0])**2))

print(list(alternatives_predictions_float.items())[:10])

with open(f"/u/scr/mhahn/PRETRAINED/SyntaxGym/syntaxgym_259_alternatives.tsv", "r") as inFile:
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
   verb = original[original.index("@")+2:-5]
   #print(verb)
   original = original[:original.index("@")] + "VERB"
   #print(original)
   #quit()
#   print(original)
   tokenized = alternative[2].split(" ")
 #  print(tokenized)
   for variant in alternative[3:]:
      #print(variant)
      if len(variant) < 5:
         continue

      subset, sentence= variant.strip().split("\t")
      sentence = "".join(sentence.strip().split(" "))
      sentence = sentence.replace("▁", " ").replace("</s>", "")
      sentence = sentence.strip()
      sentence = sentence[:-len(verb)] + "VERB"
      if sentence not in alternatives_predictions_float:
         print("DID NOT FIND", sentence)
      #   assert False
         continue
      assert sentence in alternatives_predictions_float, sentence


      variants_set[original].add(sentence)
      if subset not in variants_dict[original]:
         variants_dict[original][subset] = []
      variants_dict[original][subset].append(sentence)
  # print((result))
#   print(len(variants_set[original]), "variants")
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

with open(f"/u/scr/mhahn/sensitivity/sensitivities/sensitivities_{__file__}", "w") as outFile:
 print("Original", "\t", "BinarySensitivity", file=outFile)
 for original in variants_dict:
   varianceBySubset = {}
   assert len(variants_dict[original]) > 0
   print("=====", original)
   for subset in variants_dict[original]:
       print("...", subset)
       values = torch.FloatTensor([ valuesPerVariant[x] for x in variants_dict[original][subset]])
       #print(subset, mean(values), variance(values))
       varianceBySubset[subset] = variance(values)
       print([(x, valuesPerVariant[x]) for x in variants_dict[original][subset]])
       print(varianceBySubset[subset])



   subsetsEnumeration = list(variants_dict[original])
   if len(subsetsEnumeration) == 0:
     continue 
   N = len(subsetsEnumeration[0])
   A = [[0 for subset in range(len(subsetsEnumeration))] for inp in range(N)]
   for inp in range(N):
       for subset, bitstr in enumerate(subsetsEnumeration):
     #     print(bitstr, N)
#          assert len(bitstr) == N # This may not be satisfied here because of the way this task (SyntaxGym 259) is created
          if inp < len(bitstr) and  bitstr[inp] == "1":
              A[inp][subset] = 1
   
   
   b = [1 for _ in range(N)]
   x_bounds = [(0,1) for _ in range(len(subsetsEnumeration))]
   perSubsetSensitivities = [varianceBySubset[x] for x in subsetsEnumeration]

   sensitivity = getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities)
   print(original)
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


