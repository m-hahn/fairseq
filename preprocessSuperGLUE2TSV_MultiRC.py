import sys
TASK =  "MultiRC"
import json
for partition in ["train", "val"]:
  columns = None
  with open(f"/u/scr/mhahn/PRETRAINED/SuperGLUE/{TASK}/{partition}.jsonl", "r") as inFile:
   with open(f"/u/scr/mhahn/PRETRAINED/SuperGLUE/{TASK}/{partition if partition != 'val' else 'dev'}.tsv", "w") as outFile:
    for line in inFile:
       line = json.loads(line.strip())
       if columns is None:
          columns = ["idx", "text", "question", "answer", "label", "version"]
          print("\t".join(columns), file=outFile)
       assert columns is not None
       passage = line["passage"]
       idx = line["idx"]
       version = line["version"]
       for question in passage["questions"]:
          quest_text = question["question"]
          for answer in question["answers"]:
             idx_a = answer["idx"]
             answer_text = answer["text"]
             label = answer["label"]
             
             print("\t".join([str(idx)+"_"+str(idx_a), passage["text"], quest_text, answer_text, str(label), str(version)]), file=outFile)
