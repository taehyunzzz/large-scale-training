BATCH_SIZE=1
BACKPROP_THRU_TIME=128
EMBEDDING_SIZE=512
NUM_LAYERS=1
HIDDEN_SIZE=16
NUM_HEADS=8
DROPOUT=0.1
DATASET="wikitext2"
NUM_EPOCHS=10
RESUME="False"
DEEPSPEED_CONFIG_FILE="ds_config.json"

SUFFIX="_1st"
NO_WRITE_RESULTS=$1

for BATCH_SIZE in 1
#for BATCH_SIZE in 1 4 8 16 32
#for BATCH_SIZE in 32
do
for BACKPROP_THRU_TIME in 128 #512 
#for BACKPROP_THRU_TIME in 512 #256
#for BACKPROP_THRU_TIME in 16 64 256 
do
#for EMBEDDING_SIZE in 512
for EMBEDDING_SIZE in 64 #1024
#for EMBEDDING_SIZE in 64 128 256
do
for NUM_LAYERS in 1
#for NUM_LAYERS in 24
#for NUM_LAYERS in 1 16 128
do
for HIDDEN_SIZE in 4096
#for HIDDEN_SIZE in 4096
#for HIDDEN_SIZE in 16 64 256
do
for NUM_HEADS in 16
#for NUM_HEADS in 16
#for NUM_HEADS in 8 16 32
do
for mode in 3
#for mode in 0 3
do 

for cpu_offload_param in 0
#for cpu_offload_param in 0 1
do

#for cpu_offload_optim in 0 
for cpu_offload_optim in 1
do

if [[ ${mode} -eq 0 ]]; then
    if [[ ${cpu_offload_param} -eq 1 ]]; then
        continue
        fi
    if [[ ${cpu_offload_optim} -eq 1 ]]; then
        continue
        fi
    fi

echo "Running deepspeed mode batch${BATCH_SIZE}, bptt${BACKPROP_THRU_TIME}, emsize${EMBEDDING_SIZE}, layers${NUM_LAYERS}, hidden${HIDDEN_SIZE}, mode${mode}" cpu_offload ${cpu_offload_param} , ${cpu_offload_optim}

if [[ ${NO_WRITE_RESULTS} -eq 0 ]]; then
    deepspeed \
    --master_port $((8889+${NO_WRITE_RESULTS})) \
    --include localhost:${NO_WRITE_RESULTS} \
    train.py \
    --batch_size ${BATCH_SIZE} \
    --bptt ${BACKPROP_THRU_TIME} \
    --emsize ${EMBEDDING_SIZE} \
    --layers ${NUM_LAYERS} \
    --hidden ${HIDDEN_SIZE} \
    --heads ${NUM_HEADS} \
    --dropout ${DROPOUT} \
    --dataset ${DATASET} \
    --epochs ${NUM_EPOCHS} \
    --resume ${RESUME} \
    --cpu_offload_param ${cpu_offload_param} \
    --cpu_offload_optim ${cpu_offload_optim} \
    --fp16 \
    --suffix ${SUFFIX} \
    --deepspeed \
    --deepspeed_config "ds_config"${mode}".json"
else
    deepspeed \
    --master_port $((8889+${NO_WRITE_RESULTS})) \
    --include localhost:${NO_WRITE_RESULTS} \
    train.py \
    --batch_size ${BATCH_SIZE} \
    --bptt ${BACKPROP_THRU_TIME} \
    --emsize ${EMBEDDING_SIZE} \
    --layers ${NUM_LAYERS} \
    --hidden ${HIDDEN_SIZE} \
    --heads ${NUM_HEADS} \
    --dropout ${DROPOUT} \
    --dataset ${DATASET} \
    --epochs ${NUM_EPOCHS} \
    --resume ${RESUME} \
    --cpu_offload_param ${cpu_offload_param} \
    --cpu_offload_optim ${cpu_offload_optim} \
    --fp16 \
    --suffix ${SUFFIX} \
    --no_write_results \
    --deepspeed \
    --deepspeed_config "ds_config"${mode}".json"
    fi


done
done 
done 
done
done
done
done
done
done
