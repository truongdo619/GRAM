source ~/.bashrc
  
REPO_PATH=~/GRAM

conda activate GRAM_env && cd $REPO_PATH && export PYTHONPATH="$PYTHONPATH:$REPO_PATH" && \

DATA_DIR=~/GRAM/data/top_v2

BERT_DIR=roberta-large

LR=1e-5
BERT_DROPOUT=0.2
MRC_DROPOUT=0.5
WARMUP=1000
MAXLEN=512
MAXNORM=1.0
INTER_HIDDEN=1024

BATCH_SIZE=16
PREC=16
VAL_CKPT=1.0
ACC_GRAD=1
MAX_EPOCH=50
PROGRESS_BAR=1
OPTIM=adam
EMD_ATTN_DROPOUT=0.2
FREEZE_EPOCHS=5

for c in 0 1 2
do 
    OUTPUT_DIR=~/GRAM/outputs/GRAM_top_v2_reminder_500spis_seed_$c
    mkdir -p ${OUTPUT_DIR}
    mkdir -p ${OUTPUT_DIR}/dev_logs
    mkdir -p ${OUTPUT_DIR}/test_logs

    CUDA_VISIBLE_DEVICES=0 python ${REPO_PATH}/trainers/GRAM_trainer.py \
    --gpus="1" \
    --distributed_backend=ddp \
    --workers 0 \
    --data_dir ${DATA_DIR} \
    --bert_config_dir ${BERT_DIR} \
    --max_length ${MAXLEN} \
    --batch_size ${BATCH_SIZE} \
    --precision=${PREC} \
    --progress_bar_refresh_rate ${PROGRESS_BAR} \
    --lr ${LR} \
    --val_check_interval ${VAL_CKPT} \
    --accumulate_grad_batches ${ACC_GRAD} \
    --default_root_dir ${OUTPUT_DIR} \
    --mrc_dropout ${MRC_DROPOUT} \
    --max_epochs ${MAX_EPOCH} \
    --warmup_steps ${WARMUP} \
    --gradient_clip_val ${MAXNORM} \
    --optimizer ${OPTIM} \
    --classifier_intermediate_hidden_size ${INTER_HIDDEN} \
    --bert_dropout ${BERT_DROPOUT} \
    --label_emb_attn_dropout ${EMD_ATTN_DROPOUT} \
    --seed $c \
    --domain reminder \
    --num_freeze_bert_epochs ${FREEZE_EPOCHS} \
    --spis 500
done


wait
echo "All done"