from fairseq.models.roberta import RobertaModel
from examples.roberta.wsc import wsc_utils  # also loads WSC task and criterion
from examples.roberta.wsc import wsc_task
#roberta = RobertaModel.from_pretrained('checkpoints', 'checkpoint_best.pt', 'WSC/')
roberta = RobertaModel.from_pretrained('/u/scr/mhahn/PRETRAINED/roberta.large.wsc', "model.pt", "/juicier/scr120/scr/mhahn/PRETRAINED/WSC/")
roberta.cuda()
nsamples, ncorrect = 0, 0
for sentence, label in wsc_utils.jsonl_iterator('/juicier/scr120/scr/mhahn/PRETRAINED/WSC/val.jsonl', eval=True):
    pred = roberta.disambiguate_pronoun(sentence)
    print(sentence)
    print(pred)
    nsamples += 1
    if pred == label:
        ncorrect += 1
    break
print('Accuracy: ' + str(ncorrect / float(nsamples)))
# Accuracy: 0.9230769230769231
