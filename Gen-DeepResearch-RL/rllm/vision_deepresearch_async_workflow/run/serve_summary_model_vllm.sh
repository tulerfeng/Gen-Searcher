
echo "[Serve] Host IP: $(hostname -I 2>/dev/null | awk '{print $1}' || hostname 2>/dev/null || echo 'unknown')"
echo "[Serve] Serve URL: http://$(hostname -I 2>/dev/null | awk '{print $1}'):8001/v1"
echo "[Serve] ============================="

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve \
  Qwen/Qwen3-VL-30B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.8 \
  --served-model-name "Qwen3-VL-30B-A3B-Instruct" \
  --max-model-len 160000 \
  --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching
