TASK = "CB"
import json
for partition in ["train", "val"]:
  columns = None
  with open(f"/u/scr/mhahn/PRETRAINED/SuperGLUE/{TASK}/{partition}.jsonl", "r") as inFile:
   with open(f"/u/scr/mhahn/PRETRAINED/SuperGLUE/{TASK}/{partition if partition != 'val' else 'dev'}.tsv", "w") as outFile:
    for line in inFile:
       line = json.loads(line.strip())
       if columns is None:
          columns = sorted(list(line))
          print("\t".join(columns), file=outFile)
       assert columns is not None
       print("\t".join([str(line[x]) for x in columns]), file=outFile)
