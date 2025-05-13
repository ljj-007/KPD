BASE_PATH="."

export TF_CPP_MIN_LOG_LEVEL=3


# llama3
# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/prompt \
    --model-path ${BASE_PATH}/checkpoints/llama-3.1-8b \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --only-prompt \
    --model-type llama3

# llama3
# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/dolly/ \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/full \
    --model-path ${BASE_PATH}/checkpoints/llama-3.1-8b \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type llama3