from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    'checkpoints_CB/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='CB-bin'
)

import torch
label_fn = lambda label: roberta.task.label_dictionary.string(
    torch.LongTensor([label + roberta.task.label_dictionary.nspecial])
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
evaluatedSoFar = set()


premiseToHypothesis = {}
with open('/u/scr/mhahn/PRETRAINED/SuperGLUE/CB/dev.tsv') as fin:
   next(fin)
   for line in fin:
     line = line.split("\t")
     assert line[3] not in premiseToHypothesis
     premiseToHypothesis[line[3]] = line[0]


with open('/u/scr/mhahn/PRETRAINED/SuperGLUE/CB/dev_alternatives_c.tsv') as fin:
  with open('/u/scr/mhahn/PRETRAINED/SuperGLUE/CB/dev_alternatives_predictions_fairseq.tsv', "w") as outFile:
    while True:
        line = next(fin).strip()
        if line == "#####":
           original = next(fin) # the original
           hypothesis = premiseToHypothesis[original]
           separation = int(next(fin).strip()) # position of separation
           next(fin)
           line = next(fin)

#           tokens = roberta.encode(original, hypothesis)
#           prediction = roberta.predict('sentence_classification_head', tokens)
#           prediction_label = label_fn(prediction.argmax().item())
#           prediction = [float(x) for x in prediction.view(-1)]
#           print(original, hypothesis, prediction)
#           continue
#        continue
        #print(line)
        subset, sentences = line.strip().split("\t")
        sentences = sentences.strip().split(" ")
        sentences = "".join(sentences)
        sentences = sentences.replace("▁", " ")
        sentences = sentences.strip()
        if sentences in evaluatedSoFar:
           continue
        evaluatedSoFar.add(sentences)
        if len(evaluatedSoFar) % 100 == 0:
           print(sentences)
        tokens = roberta.encode(sentences, hypothesis)
        prediction = roberta.predict('sentence_classification_head', tokens)
        prediction_label = label_fn(prediction.argmax().item())
        prediction = [float(x) for x in prediction.view(-1)]
        print(prediction)
        print("\t".join([sentences, hypothesis, " ".join([str(x) for x in prediction])]), file=outFile)

