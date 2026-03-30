
echo "[Serve] Host IP: $(hostname -I 2>/dev/null | awk '{print $1}' || hostname 2>/dev/null || echo 'unknown')"
echo "[Serve] Serve URL: http://$(hostname -I 2>/dev/null | awk '{print $1}'):8001/v1"
echo "[Serve] ============================="

CUDA_VISIBLE_DEVICES=0 vllm serve \
  GenSearcher/Gen-Searcher-8B \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.8 \
  --served-model-name "Gen-Searcher-8B" \
  --max-model-len 160000 \
  --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching
