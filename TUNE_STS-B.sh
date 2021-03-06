TOTAL_NUM_UPDATES=3598  # 10 epochs through STS-B for bsz 16
WARMUP_UPDATES=214      # 6 percent of the number of updates
LR=2e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=1
MAX_SENTENCES=16        # Batch size.
ROBERTA_PATH=/u/scr/mhahn/PRETRAINED/roberta.large.mnli/model.pt

~/python-py37-mhahn train.py STS-B-bin/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --save-dir /jagupard12/scr2/mhahn/checkpoints_STS-B \
    --best-checkpoint-metric accuracy --regression-target --best-checkpoint-metric loss;
