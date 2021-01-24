from fairseq.models.roberta import RobertaModel
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()


import sys

model = sys.argv[1]
assert model in ["cr", "mr", "mpqa", "subj"]

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
with open(f'/u/scr/mhahn/PRETRAINED/textclas/{model}_alternatives_predictions_PMLM_1billion_fairseq.tsv', "w") as outFile:
 for group in [""]:
   with open(f'/u/scr/mhahn/PRETRAINED/textclas/{model}_alternatives_PMLM_1billion_raw.tsv', "r") as fin:
    try:
     while True:
        line = next(fin).strip()
        try:
           subset, original_tokenized, alternative = line.strip().split("\t")
        except ValueError:
           print("ValueError: ", line)
           continue

        # This is for cutting of stuff after the SEP.
        alternative = alternative.split("[SEP]")
        assert len(alternative) >= 1, alternative
        assert len(alternative[0]) > 5, alternative
        assert "[CLS]" in alternative[0], alternative
        alternative = alternative[0]


        alternativeOriginal = alternative.strip()

        alternative = alternative.replace("[CLS]", "").replace("[SEP]", "").strip().replace(" ' s ", " 's ").replace(" ' ll ", " 'll ").replace(" ' d ", " 'd ").replace("n ' t ", "n't ").replace(" ' ve ", " 've ").replace(" @ - @ ", "-").replace("( ", "(")
        alternative = detokenizer.detokenize(alternative.split(" "))

        sentences = [alternative]
        if alternativeOriginal in evaluatedSoFar:
           continue
        evaluatedSoFar.add(alternativeOriginal)
        if len(evaluatedSoFar) % 100 == 0:
           print(len(evaluatedSoFar), sentences)
        tokens = roberta.encode(sentences[0])
        prediction = roberta.predict('sentence_classification_head', tokens)
        prediction_label = label_fn(prediction.argmax().item())
        prediction = [float(x) for x in prediction.view(-1)]
        print("\t".join([alternativeOriginal, str(prediction[1] if len(prediction) == 2 else " ".join(str(y) for y in prediction)), prediction_label]), file=outFile)
    except StopIteration:
      pass     

