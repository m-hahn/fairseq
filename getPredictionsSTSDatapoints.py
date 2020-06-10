from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    'checkpoints_STS-B/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='STS-B-bin'
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
with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/STS-B/dev_alternatives_c.tsv', "r") as fin:
  with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/STS-B/dev_datapoints_predictions_fairseq.tsv', "w") as outFile:
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
        sentences = [sentences[:separation], sentences[separation:]]
        assert len(sentences[1]) > 1, (line, separation, sentences)
        for i in range(2):
          sentences[i] = ("".join(sentences[i])).replace("▁", " ").replace("</s>", "").strip()
        assert len(sentences[1]) > 1, (line, separation, sentences)
        assert sentences[0].endswith("."), (line, separation, sentences)
#        print(sentences)
        if tuple(sentences) in evaluatedSoFar:
           continue
        evaluatedSoFar.add(tuple(sentences))
        if len(evaluatedSoFar) % 100 == 0:
           print(len(evaluatedSoFar), sentences)
        tokens = roberta.encode(sentences[0], sentences[1])

# https://github.com/pytorch/fairseq/issues/1009
        features = roberta.extract_features(tokens)
        prediction = float(5.0 * roberta.model.classification_heads['sentence_classification_head'](features))
        print("\t".join([sentences[0], sentences[1], str(prediction)]), file=outFile)

