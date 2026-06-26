# Continual self-learning container (GPU). Build once, run anywhere with --gpus all.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends git espeak-ng \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir transformers trl datasets accelerate panphon wordfreq numpy
ARG REPO=grundlagen/Lingua-Sound-Wave
ARG BRANCH=claude/phrase-weave-multiword
RUN git clone --depth 1 -b ${BRANCH} https://github.com/${REPO}.git /lingua
WORKDIR /lingua/research/homophone-bench/selflearn
ENV PYTHONUNBUFFERED=1 CKPT=/ckpt
VOLUME /ckpt
# loops forever, auto-restarts rounds; --eval_llm activates if OPENROUTER_API_KEY set
CMD ["bash","-lc","python3 run_continual.py --base ${BASE:-Qwen/Qwen2.5-1.5B-Instruct} --k 8 --keep_thresh 0.55 --ckpt_dir $CKPT ${OPENROUTER_API_KEY:+--eval_llm}"]
