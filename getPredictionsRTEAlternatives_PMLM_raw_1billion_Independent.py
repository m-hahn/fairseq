from fairseq.models.roberta import RobertaModel
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()


import sys

model = sys.argv[1]
assert model == "RTE"

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
with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_alternatives_predictions_PMLM_1billion_raw_Independent.tsv', "w") as outFile:
 with open(f'/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_alternatives_PMLM_1billion_raw_Independent.tsv', 'r') as fin:
    while True:
        line = next(fin).strip()
        try:
           subset, original_tokenized, alternative = line.strip().split("\t")
        except ValueError:
           print("ValueError: ", line)
           continue

        alternativeOriginal = alternative.strip()

        alternatives = alternative.replace("[CLS]", "").replace("[ CLS]", "").replace("[ CLS ]", "").split("[SEP]")
        assert len(alternatives) > 1, alternatives
        if len(alternatives) > 3 or (len(alternatives) > 2 and len(alternatives[2]) > 5):
            print("ODD Text after the end:", alternatives)
        alternatives = alternatives[:2]
        for i in range(2):
           alternatives[i] = alternatives[i].replace("[CLS]", "").replace("[SEP]", "").strip().replace(" ' s ", " 's ").replace(" ' ll ", " 'll ").replace(" ' d ", " 'd ").replace("n ' t ", "n't ").replace(" ' ve ", " 've ").replace(" @ - @ ", "-").replace("( ", "(")
           alternatives[i] = detokenizer.detokenize(alternatives[i].split(" "))

                                                                                                         
        
        sentences = alternatives
        if alternativeOriginal in evaluatedSoFar:
           continue
        evaluatedSoFar.add(alternativeOriginal)
        if len(evaluatedSoFar) % 100 == 0:
           print(len(evaluatedSoFar), sentences)
        tokens = roberta.encode(sentences[0], sentences[1])
        prediction = roberta.predict('sentence_classification_head', tokens)
        prediction_label = label_fn(prediction.argmax().item())
        prediction = [float(x) for x in prediction.view(-1)]
        print("\t".join([alternativeOriginal, str(prediction[1]), {"not_entailment" : "0", "entailment" : "1"}[prediction_label]]), file=outFile)

