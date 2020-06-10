from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    'checkpoints_QNLI/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='QNLI-bin'
)

import torch
label_fn = lambda label: roberta.task.label_dictionary.string(
    torch.LongTensor([label + roberta.task.label_dictionary.nspecial])
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
evaluatedSoFar = set()
lineNumbers = 0
with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/QNLI/dev_datapoints_predictions_fairseq.tsv', "w") as outFile:
 for group in ["c", "d"]:
  with open(f'/u/scr/mhahn/PRETRAINED/GLUE/glue_data/QNLI/dev_alternatives_{group}.tsv', "r") as fin:
   try:
    while True:
        lineNumbers += 1
        try:
           line = next(fin).strip()
        except UnicodeDecodeError:
           print("UnicodeDecodeError", lineNumbers)
           continue
        except StopIteration:
           break
        if line == "#####":
           originalSentences = next(fin).strip() # the original
           separation = int(next(fin).strip()) # position of separation
           sentences = next(fin).strip().split(" ")
        else:
            continue
        print(sentences, separation)
        sentences = [sentences[:separation], sentences[separation:]]
        assert len(sentences[1]) > 3, (line, separation, sentences)
        for i in range(2):
          sentences[i] = ("".join(sentences[i])).replace("▁", " ").replace("</s>", "").strip()
        assert len(sentences[1]) > 0, (line, separation, sentences)
#        print(sentences)
        if tuple(sentences) in evaluatedSoFar:
           continue
        evaluatedSoFar.add(tuple(sentences))
        if len(evaluatedSoFar) % 100 == 0:
           print(len(evaluatedSoFar), sentences)
        tokens = roberta.encode(sentences[0], sentences[1])
        prediction = roberta.predict('sentence_classification_head', tokens)
        prediction_label = label_fn(prediction.argmax().item())
        prediction = [float(x) for x in prediction.view(-1)]
        print("\t".join([sentences[0], sentences[1], str(prediction[1]), str(prediction_label)]), file=outFile)

