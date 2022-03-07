
CKPT="checkpoints/set1/model1.ckpt"
TEST_DIR="test_dir"
OUT_DIR="test_dir/Segmentation"
NUM_PARTS=2

python predict.py \
--ckpt ${CKPT} \
--test_dir ${TEST_DIR} \
--out_dir ${OUT_DIR} \
--bb 0 0 0 3392 4992 304 304 \
--num_parts ${NUM_PARTS}