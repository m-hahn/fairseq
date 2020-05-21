from fairseq.models.roberta import RobertaModel


import sys

model = sys.argv[1]
assert model == "CoLA"

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
predictions = []
targets = []
with open(f'/u/scr/mhahn/PRETRAINED/GLUE/glue_data/{model}/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        _, target, _, sent = tokens
        tokens = roberta.encode(sent)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        print(target, prediction_label)
        predictions.append(float(prediction_label))
        targets.append(float(target))
        ncorrect += int(prediction_label == target)
        nsamples += 1
#        if nsamples == 10:
 #         break
print(nsamples)
print('| Accuracy: ', float(ncorrect)/float(nsamples))
import scipy.stats
print(scipy.stats.pearsonr(predictions, targets))
