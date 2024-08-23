FROM intelanalytics/ipex-llm-serving-vllm-xpu-experiment:2.1.0b2

COPY comps/llms/text-generation/vllm/docker/start_qwen32b.sh /llm

RUN chmod +x /llm/start_qwen32b.sh

ENTRYPOINT ["/llm/start_qwen32b.sh"]
