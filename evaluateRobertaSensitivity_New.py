"""
Evaluate average block sensitivity for WSC.
"""
import traceback

from fairseq.models.roberta import RobertaModel
from examples.roberta.wsc import wsc_utils  # also loads WSC task and criterion
from examples.roberta.wsc import wsc_task
#roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'WSC/')
nsamples, ncorrect = 0, 0

def mean(values):
   return sum(values)/len(values)

def variance(values):
   return mean([x**2 for x in values]) - mean(values)**2

roberta = RobertaModel.from_pretrained('/u/scr/mhahn/PRETRAINED/roberta.large.wsc', "model.pt", "/juicier/scr120/scr/mhahn/PRETRAINED/WSC/")
roberta.cuda()

#print(3)
#roberta.disambiguate_pronoun('Did not think that he had done anything wrong. Do not consider that he did not consider that he had done anything wrong, if _anyone_ did believe that anyone did not consider that he,  [him] did.')
#roberta.disambiguate_pronoun('did not think that he had done anything wrong. do not consider that he did not consider that he had done anything wrong, if _anyone_ did believe that anyone did not consider that he,  [him].')
#roberta.disambiguate_pronoun('“...did not think that he had done anything wrong. “...do not consider that he” did not consider that he had done anything wrong, if" _anyone_ did" believe that anyone “did not consider that he", “ [him]')
#roberta.disambiguate_pronoun('“...did not think that he had done anything wrong. “...do not consider that he” did not consider that he had done anything wrong, if" _anyone_ did" believe that anyone “did not consider that he", “ [him].')
#
#print(1)
#roberta.disambiguate_pronoun('...did not think that he had done anything wrong. ...do not consider that he” did not consider that he had done anything wrong, if _anyone_ did believe that anyone did not consider that he,  [him] ')
#print(2)
#roberta.disambiguate_pronoun('“...did not think that he had done anything wrong. “...do not consider that he” did not consider that he had done anything wrong, if _anyone_ did" believe that anyone “did not consider that he, “ [him] ')
#print(3)
#roberta.disambiguate_pronoun('“...did not think that he had done anything wrong. “...do not consider that he” did not consider that he had done anything wrong, if" _anyone_ did" believe that anyone “did not consider that he", “ [him] ')
#quit()

from scipy.optimize import linprog


def getMaxOverPartitions(A, b, x_bounds, perSubsetSensitivities):
   #print(perSubsetSensitivities)
   c = [-x for x in perSubsetSensitivities]
   res = linprog(c, A_ub=A, b_ub=b, bounds=x_bounds)
   return -res.fun

from random import shuffle

with open("/juicier/scr120/scr/mhahn/PRETRAINED/WSC/val_alternatives.txt", "r") as inFile: # _SECOND_VERSION_BACKUP
  alternatives = inFile.read().strip().split("#####\n")
  print(len(alternatives))

shuffle(alternatives)

sensitivities = []

for alternative in alternatives:
   if len(alternative) < 5:
      continue
   variants_set = set()
   variants_dict = {}
   
   alternative = alternative.split("\n")
   original = alternative[0].strip().replace(" [ ", " [")
   print(original)
   start_underscore = original.find("_")
   end_underscore = original.find("_",start_underscore+1)
   assert start_underscore < end_underscore
   assert original[start_underscore+1] == " "
   assert original[end_underscore-1] == " "
   original = original[:start_underscore] + "_" + original[start_underscore+2:end_underscore-1] + "_" + original[end_underscore+1:]
   print(original)
   print(roberta.disambiguate_pronoun(original))

   start_underscore, end_underscore, start_bracket, end_bracket = [int(x) for x in alternative[1].split(" ")]
   tokenized = alternative[2].split(" ")
   for variant in alternative[3:]:
      if len(variant) < 5:
         continue
      subset, sentence = variant.split("\t")
      if subset not in variants_dict:
         variants_dict[subset] = []
      sentence = sentence.replace("[", "").replace("]", "").replace("_", "").split(" ")
      sentence[start_underscore+1] = "_1"+sentence[start_underscore+1]
      sentence[start_bracket+1] = "["+sentence[start_bracket+1]
      sentence[end_underscore+1] = "_2 "+sentence[end_underscore+1]
      sentence[end_bracket+1] = "] "+sentence[end_bracket+1]
      sentence = (" ".join(sentence)).split(" ")
      #print(tokenized[start_underscore:end_underscore])
     # print(tokenized[start_bracket:end_bracket])
 
    #  print(sentence)
      result = [""]
      for word in sentence:
         if word.startswith("▁"):
             result.append(word[1:])
         else:
             result[-1] = result[-1] + word
      result = " ".join(result)
      #print(result)
      result = result.replace("]", "] ")
      result = result.replace("[▁", " [")
      result = result.replace("_1▁", " _")
      result = result.replace("_2", "_ ")
      result = result.replace("  ", " ")
      result = result.replace(" .", ".") # in case the sentence ends with "[PRONOUN] + punctuation"
      result = result.strip()
      #print(result)
#      assert False
      variants_set.add(result)
      variants_dict[subset].append(result)
  # print((result))
   print(len(variants_set), "variants")
   valuesPerVariant = {}
   for variant in variants_set:
   #  print(variant)
     try:
       valuesPerVariant[variant] = 1 if roberta.disambiguate_pronoun(variant) == True else -1
       if len(valuesPerVariant) % 100 == 0:
         print(valuesPerVariant[variant], valuesPerVariant[variant] == True, len(valuesPerVariant), len(variants_set), variant)
     except ValueError:
        print("VALUE ERROR", variant)
        valuesPerVariant[variant] = 0
     except AttributeError:
        print("ATTRIBUTE ERROR", variant)
        valuesPerVariant[variant] = 0
     except RuntimeError:
        print("RUNTIME ERROR", variant)
        valuesPerVariant[variant] = 0
     except Exception as e:
        print("EXCEPTION ", variant)
        print(traceback.print_exc())
        print(e)
        valuesPerVariant[variant] = 0
   varianceBySubset = {}
   for subset in variants_dict:
       values = [ valuesPerVariant[x] for x in variants_dict[subset]]
       print(subset, mean(values), variance(values))
       varianceBySubset[subset] = variance(values)
   print(varianceBySubset)


   subsetsEnumeration = list(variants_dict)
    
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
   sensitivities.append(sensitivity)
   print("Average sensitivity so far", sum(sensitivities)/len(sensitivities))


