python3 -m eval.eval_runner \
  --parquet /path/to/test.parquet \
  --base-url http://10.144.200.237:8001/v1 \
  --model Vision-DeepResearch-8B \
  --parallel-tasks 5 \
  --api-key EMPTY \
  2>&1 | tee eval/log.txt

