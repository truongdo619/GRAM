source ~/.bashrc
  
REPO_PATH=~/GRAM

conda activate GRAM_env && cd $REPO_PATH && export PYTHONPATH="$PYTHONPATH:$REPO_PATH" && \

DATA_DIR=~/GRAM/data/atis

LR=1e-5
BERT_DROPOUT=0.2
MRC_DROPOUT=0.5
WARMUP=1000
MAXLEN=512
MAXNORM=1.0
INTER_HIDDEN=512

BATCH_SIZE=32
PREC=16
ACC_GRAD=1
MAX_EPOCH=130
OPTIM=adam
EMD_ATTN_DROPOUT=0.5
FREEZE_EPOCH=30
NUM_HEADS=1

for c in 0 1 2
do 
    BERT_DIR=roberta-large
    echo $BERT_DIR

    OUTPUT_DIR=~/GRAM/outputs/GRAM_atis_seed_$c
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
    --lr ${LR} \
    --accumulate_grad_batches ${ACC_GRAD} \
    --default_root_dir ${OUTPUT_DIR} \
    --mrc_dropout ${MRC_DROPOUT} \
    --max_epochs ${MAX_EPOCH} \
    --span_loss_candidates ${SPAN_CANDI} \
    --warmup_steps ${WARMUP} \
    --gradient_clip_val ${MAXNORM} \
    --optimizer ${OPTIM} \
    --classifier_intermediate_hidden_size ${INTER_HIDDEN} \
    --bert_dropout ${BERT_DROPOUT} \
    --num_freeze_bert_epochs ${FREEZE_EPOCH} \
    --label_emb_attn_dropout ${EMD_ATTN_DROPOUT} \
    --label_emb_attn_num_heads ${NUM_HEADS} \
    --seed $c
done

wait
echo "All done"