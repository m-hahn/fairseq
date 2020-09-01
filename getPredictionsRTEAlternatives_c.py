from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    'checkpoints_RTE/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='RTE-bin'
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
with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_alternatives_c.tsv', "r") as fin:
  with open('/u/scr/mhahn/PRETRAINED/GLUE/glue_data/RTE/dev_alternatives_c_predictions_fairseq.tsv', "w") as outFile:
    while True:
        lineNumbers += 1
        try:
           line = next(fin).strip()
        except UnicodeDecodeError:
           print("UnicodeDecodeError", lineNumbers)
           continue
        if line == "#####":
           next(fin) # the original
           separation = int(next(fin).strip()) # position of separation
           next(fin)
           line = next(fin)
        #print(line)
        subset, sentences = line.strip().split("\t")
        sentences = sentences.strip().split(" ")
        sentences = [sentences[:separation], sentences[separation:]]
        for i in range(2):
          sentences[i] = "".join(sentences[i])
          sentences[i] = sentences[i].replace("▁", " ")
          if "<" in sentences[i]:
            sentences[i] = sentences[i][sentences[i].rfind("<")+1:]
          if ">" in sentences[i]:
            sentences[i] = sentences[i][sentences[i].rfind(">")+1:]
          sentences[i] = sentences[i].strip()
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
        print("\t".join([sentences[0], sentences[1], str(prediction[1]), {"not_entailment" : "0", "entailment" : "1"}[prediction_label]]), file=outFile)
  except StopIteration:
     pass    
