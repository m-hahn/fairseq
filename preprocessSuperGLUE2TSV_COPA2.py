import sys
TASK="COPA"
import json
for partition in ["train", "val"]:
  columns = None
  with open(f"/u/scr/mhahn/PRETRAINED/SuperGLUE/{TASK}/{partition}.jsonl", "r") as inFile:
   with open(f"/u/scr/mhahn/PRETRAINED/SuperGLUE/COPA2/{partition if partition != 'val' else 'dev'}.tsv", "w") as outFile:
    for line in inFile:
       line = json.loads(line.strip())
       if columns is None:
          columns = ["choices", "premise", "idx", "label"]
          print("\t".join(columns), file=outFile)
       assert columns is not None
       print("\t".join([line["choice1"]+" </s> </s> "+line["choice2"], line["premise"]+" </s> </s> "+line["question"], str(line["idx"]), str(line["label"])]), file=outFile)
