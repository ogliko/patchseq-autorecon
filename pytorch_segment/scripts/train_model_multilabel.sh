CKPT="None"
CKPT_DIR="run1"
LOG_DIR="logs"
JSON_DIR="data/set1"
EPS=1e-1
EPOCHS=100
NUM_STACKS=3
AUGMENTATION=1
PIA_DIR="data/set_pia"

python train_multilabel.py \
--ckpt ${CKPT} \
--ckpt_dir ${CKPT_DIR} \
--log_dir ${LOG_DIR} \
--json_dir ${JSON_DIR} \
--eps ${EPS} \
--epochs ${EPOCHS} \
--num_stacks ${NUM_STACKS} \
--augmentation ${AUGMENTATION} \
--pia_dir ${PIA_DIR}