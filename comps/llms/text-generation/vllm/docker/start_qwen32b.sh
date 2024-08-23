#!/bin/bash
model=/llm/models
served_model_name="Qwen1.5-32B-Chat"

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
export TORCH_LLM_ALLREDUCE=0
export CCL_DG2_ALLREDUCE=1

# Tensor parallel related arguments:
export CCL_WORKER_COUNT=4
export FI_PROVIDER=shm
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_ATL_SHM=1

source /opt/intel/oneapi/setvars.sh
source /opt/intel/1ccl-wks/setvars.sh

python -m ipex_llm.vllm.xpu.entrypoints.openai.api_server \
  --served-model-name $served_model_name \
  --port 9009 \
  --model $model \
  --trust-remote-code \
  --gpu-memory-utilization 0.7 \
  --device xpu \
  --dtype float16 \
  --enforce-eager \
  --load-in-low-bit fp8 \
  --max-model-len 6656 \
  --max-num-batched-tokens 6656 \
  --tensor-parallel-size 4
