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

with open("/juicier/scr120/scr/mhahn/PRETRAINED/WSC/val_alternatives.txt", "r") as inFile:
  alternatives = inFile.read().strip().split("#####\n")
  print(len(alternatives))
for alternative in alternatives:
   if len(alternative) < 5:
      continue
   variants_set = set()
   variants_dict = {}
   
   alternative = alternative.split("\n")
   original = alternative[0]
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
      sentence = sentence.split(" ")
      sentence[start_underscore+1] = "_1 "+sentence[start_underscore+1]
      sentence[start_bracket+1] = "[ "+sentence[start_bracket+1]
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
#      result = result.replace("]", "]")
      result = result.replace("[", " [")
      result = result.replace("_1", " _")
      result = result.replace("_2", "_")
      result = result.strip()
      variants_set.add(result)
      variants_dict[subset].append(result)
   print((result))
   print(len(variants_set))
   valuesPerVariant = {}
   for variant in variants_set:
     print(variant)
     valuesPerVariant[variant] = roberta.disambiguate_pronoun(variant)
     print(pred)
   varianceBySubset = {}
   for subset in variants_dict:
       values = [valuesPerVariant[x] for x in variants_dict[subset]]
       print(subset, mean(values), variance(values))
