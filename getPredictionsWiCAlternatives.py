from collections import defaultdict

alternativesPerSentence = defaultdict(list)

with open('/u/scr/mhahn/PRETRAINED/SuperGLUE/WiC/dev_alternatives_c.tsv', "r") as fin:
  try:
    while True:
        line = next(fin).strip()
#        print("LINE", line, line == "#####")
        if line == "#####":
           original = next(fin).strip() # the original
           next(fin)
           tokenized = next(fin)
           line = next(fin)
        
        subset, sentences = line.strip().split("\t")
        sentence = sentences.strip().replace(" ", "").replace("▁", " ").replace("</s>", "").strip()
        alternativesPerSentence[original].append((subset, sentence))
#        print(original+"#")
  except StopIteration:
    pass

from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    'checkpoints_WiC/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='WiC-bin'
)

import torch
label_fn = lambda label: roberta.task.label_dictionary.string(
    torch.LongTensor([label + roberta.task.label_dictionary.nspecial])
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
evaluatedSoFar = dict()
with open('/u/scr/mhahn/PRETRAINED/SuperGLUE/WiC/dev.tsv') as fin:
  next(fin) # for the header
  with open('/u/scr/mhahn/PRETRAINED/SuperGLUE/WiC/dev_alternatives_predictions_fairseq.tsv', "w") as outFile:
    while True:
        line = next(fin).strip().split("\t")
        _, _, _, _, sentence1, sentence2, _, _, _, word = line
        print("#####", file=outFile)
        print(sentence1+" "+sentence2, file=outFile)
        print(word, file=outFile)
        alternatives1 = alternativesPerSentence[sentence1]
        alternatives2 = alternativesPerSentence[sentence2]
        assert len(alternatives1) > 0
        assert len(alternatives2) > 0
        defaultSubset1 = "0"*(len(alternatives1[0][0]))
        defaultSubset2 = "0"*(len(alternatives2[0][0]))
        for i in range(2):
          for j in range(len(alternatives1 if i == 0 else alternatives2)):
              subset1 = defaultSubset1 if i == 1 else alternatives1[j][0]
              sent1 = sentence1 if i == 1 else alternatives1[j][1]
              subset2 = defaultSubset2 if i == 0 else alternatives2[j][0]
              sent2 = sentence2 if i == 0 else alternatives2[j][1]
#              print(alternatives1)
 #             print(alternatives2)
  #            quit()
              sentences = (sent1, sent2)
              if sentences in evaluatedSoFar:
                 predictions = evaluatedSoFar[sentences]
              else:
                 if len(evaluatedSoFar) % 100 == 0:
                    print(sentences)
                 tokens = roberta.encode(sentences[0], sentences[1])
                 prediction = roberta.predict('sentence_classification_head', tokens)
                 prediction_label = label_fn(prediction.argmax().item())
                 prediction = [float(x) for x in prediction.view(-1)]
                 predictions = (str(prediction[1]), prediction_label)
                 evaluatedSoFar[sentences] = predictions
              print("\t".join([subset1+subset2, sentences[0], sentences[1], predictions[0], predictions[1]]), file=outFile)
         
