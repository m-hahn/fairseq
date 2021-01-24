from fairseq.models.roberta import RobertaModel
from nltk.tokenize.treebank import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()


import sys

model = sys.argv[1]
assert model == "SST-2"

roberta = RobertaModel.from_pretrained(
    f'checkpoints_{model}/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path=f'{model}-bin'
)

import torch
label_fn = lambda label: roberta.task.label_dictionary.string(
    torch.LongTensor([label + roberta.task.label_dictionary.nspecial])
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
evaluatedSoFar = set()
with open('../Robustness-Low-Synergy-and-Cheap-Computation/experiments/202-sst2/Submiterator-master/provided-alternatives_labeled.tsv', "w") as outFile:
 with open(f'../Robustness-Low-Synergy-and-Cheap-Computation/experiments/202-sst2/Submiterator-master/provided-alternatives.tsv', 'r') as fin:
    while True:
      line = next(fin).strip()
      try:
         original, subsets, neighbor = line.strip().split("\t")
      except ValueError:
         print("ValueError: ", line)
         continue
      for alternative in [original, neighbor]:
        alternativeOriginal = alternative.strip()

        alternatives = [" ".join(tokenizer.tokenize(alternative))]
        alternatives = alternatives[:1]
        for i in range(1):
           alternatives[i] = alternatives[i].strip()

                                                                                                         
        
        sentences = alternatives
        if alternativeOriginal in evaluatedSoFar:
           continue
        evaluatedSoFar.add(alternativeOriginal)
        if len(evaluatedSoFar) % 100 == 0:
           print(len(evaluatedSoFar), sentences)
        tokens = roberta.encode(sentences[0])
        prediction = roberta.predict('sentence_classification_head', tokens)
        prediction_label = label_fn(prediction.argmax().item())
        prediction = [float(x) for x in prediction.view(-1)]
        print("\t".join([alternativeOriginal, str(prediction[1]), prediction_label]), file=outFile)
#  except StopIteration:
 #    pass
