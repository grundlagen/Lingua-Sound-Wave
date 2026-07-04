"""Self-learning homophonic carver: SFT warm-start then best-of-N self-improvement.

Designed to run on ONE consumer GPU (Colab T4 / RunPod 3090-4090). Trains a small
base model to rewrite English so it sounds like French, improving itself against
our reward (prosody sound-match x French-validity) with NO human labels.

Loop (rejection-sampling / "expert iteration" -- robust, no RL plumbing):
  0. SFT warm-start on train-homophonic.jsonl (our generated corpus).
  1. For a batch of English phrases, SAMPLE k French candidates from the model.
  2. Score each with reward.reward(en, fr)  (local, free, prosody-aware).
  3. Keep the best candidate per phrase above a threshold -> new SFT pairs.
  4. SFT on those self-generated bests. Repeat 1-4. The model bootstraps upward.

Periodically eval with the LLM judge (fr_coherence) -- a sanity check, not the
per-sample signal. This is in-weight self-learning; the reward is the teacher.

Run on a GPU box (see selflearn/README.md):
  pip install transformers trl datasets accelerate
  python train_selflearn.py --base Qwen/Qwen2.5-1.5B-Instruct --rounds 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))      # parent: fr_coherence, matcher
import reward as R
try:
    from fr_coherence import FRCoherence       # optional LLM eval (needs key)
except Exception:
    FRCoherence = None

INSTR = "Rewrite the English so it sounds the same when read aloud in French:"


def _env_versions():
    """Pinned lib versions -> logged with every crash so version-skew bugs
    (the usual trl/transformers breakage) are diagnosable from Drive alone."""
    import importlib
    out = []
    for m in ("torch", "transformers", "trl", "datasets", "accelerate"):
        try:
            out.append(f"{m}=={importlib.import_module(m).__version__}")
        except Exception:
            out.append(f"{m}==?")
    return " ".join(out)


def load_sft(path, cap=40000):
    """Old carve corpus AND train-dual-v1 rows both load; capped for T4 rounds."""
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        t = r.get("task", "")
        if t in ("phrase_carve", "word_carve") or t.startswith("dual") or \
           t in ("glue", "bridge", "phrase_unit", "line_carve"):
            if r.get("prompt") and r.get("completion"):
                rows.append((r["prompt"], r["completion"]))
    if len(rows) > cap:
        import random as _r
        _r.Random(0).shuffle(rows)
        rows = rows[:cap]
        print(f"[data] capped to {cap} rows for round tractability")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="../train-homophonic.jsonl")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--k", type=int, default=8, help="samples per phrase")
    ap.add_argument("--keep_thresh", type=float, default=0.55)
    ap.add_argument("--out", default="./homophonic-carver")
    ap.add_argument("--ckpt_dir", default="",
                    help="Drive path for checkpoint+status (survives disconnects)")
    ap.add_argument("--eval_llm", action="store_true",
                    help="score each round's samples with the Nemotron judge")
    ap.add_argument("--continual", action="store_true",
                    help="never stop: iterate rounds forever, skipping round errors")
    ap.add_argument("--sft_only", action="store_true",
                    help="pure supervised fine-tune on the GOLD corpus, then eval "
                         "and stop -- no reward loop (the robust, no-stall path)")
    ap.add_argument("--epochs", type=int, default=3,
                    help="SFT epochs for --sft_only")
    args = ap.parse_args()
    if args.data.endswith("train-homophonic.jsonl"):
        dual = os.path.join(os.path.dirname(os.path.abspath(args.data)),
                            "train-dual-v1.jsonl")
        if not os.path.exists(dual):
            try:
                import subprocess as _sp
                _sp.run([sys.executable, "build_train_corpus.py"], timeout=900,
                        cwd=os.path.dirname(dual))
            except Exception:
                pass
        if os.path.exists(dual):
            print(f"[data] auto-upgrading to the dual corpus: {dual}")
            args.data = dual
    ckpt = args.ckpt_dir or args.out
    os.makedirs(ckpt, exist_ok=True)
    status_path = os.path.join(ckpt, "status.json")

    def gh_push(text):
        """PUT status.json to a 'selflearn-status' branch so it can be monitored
        remotely. Needs GITHUB_TOKEN + GITHUB_REPO (owner/name) env vars."""
        import base64
        import urllib.request
        tok_, repo_ = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPO")
        if not (tok_ and repo_):
            return
        api = f"https://api.github.com/repos/{repo_}/contents/selflearn/status.json"
        hdr = {"Authorization": f"Bearer {tok_}", "Accept": "application/vnd.github+json"}
        sha = None
        try:
            q = urllib.request.Request(api + "?ref=selflearn-status", headers=hdr)
            sha = json.load(urllib.request.urlopen(q, timeout=20)).get("sha")
        except Exception:
            pass
        body = {"message": "selflearn status", "branch": "selflearn-status",
                "content": base64.b64encode(text.encode()).decode()}
        if sha:
            body["sha"] = sha
        try:
            req = urllib.request.Request(api, data=json.dumps(body).encode(),
                                         headers=hdr, method="PUT")
            urllib.request.urlopen(req, timeout=20)
        except Exception as e:
            print(f"[status push skipped: {e}]")

    def write_status(**kw):
        st = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
        text = json.dumps(st, ensure_ascii=False, indent=1)
        json.dump(st, open(status_path, "w"), ensure_ascii=False, indent=1)
        gh_push(text)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    # T4 has no bf16; auto-pick the supported half precision
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    # resume from a Drive checkpoint if one exists (survives Colab disconnects)
    start_round, resumed = 0, False
    src_model = args.base
    if os.path.exists(os.path.join(ckpt, "config.json")):
        src_model, resumed = ckpt, True
        if os.path.exists(status_path):
            start_round = json.load(open(status_path)).get("round", -1) + 1
        print(f"RESUMING from {ckpt} at round {start_round}")

    tok = AutoTokenizer.from_pretrained(src_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(src_model, dtype=dtype,
                                                     device_map="auto")
    except TypeError:                       # older transformers
        model = AutoModelForCausalLM.from_pretrained(src_model, torch_dtype=dtype,
                                                     device_map="auto")
    # Gradient checkpointing needs the input embeddings to require grad, else the
    # backward pass dies with "element 0 of tensors does not require grad" (the SFT
    # training_step crash). use_cache must also be off during training.
    model.config.use_cache = False
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    try:
        judge = FRCoherence() if (args.eval_llm and FRCoherence) else None
    except Exception as _e:
        print(f"[eval_llm disabled: {_e}]")
        judge = None

    def to_text(prompt, completion):
        msgs = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}]
        return tok.apply_chat_template(msgs, tokenize=False)

    def sft(pairs, epochs=1, tag=""):
        import inspect
        ds = Dataset.from_dict({"text": [to_text(p, c) for p, c in pairs]})
        # T4 (14.5GB) can't hold 1.5B + Adam fp32 states; adafactor + grad
        # checkpointing + micro-batch keeps the same effective batch of 16
        want = dict(output_dir=f"{args.out}-{tag}", num_train_epochs=epochs,
                    per_device_train_batch_size=2, gradient_accumulation_steps=8,
                    learning_rate=1e-5, logging_steps=20, save_strategy="no",
                    bf16=bf16, fp16=not bf16, optim="adafactor",
                    gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                    max_seq_length=128, max_length=128,      # trl renamed it
                    dataset_text_field="text")
        ok = set(inspect.signature(SFTConfig.__init__).parameters)
        cfg = SFTConfig(**{k: v for k, v in want.items() if k in ok})
        tkw = set(inspect.signature(SFTTrainer.__init__).parameters)
        extra = ({"processing_class": tok} if "processing_class" in tkw
                 else {"tokenizer": tok})
        try:
            SFTTrainer(model=model, args=cfg, train_dataset=ds, **extra).train()
        except Exception:
            import traceback
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "TRAIN_ERRORS.log"), "a") as ef:
                ef.write("\n=== sft() failure ===\n" + traceback.format_exc())
            raise

    def sample(prompts, k):
        outs = []
        for p in prompts:
            # transformers>=5 returns a BatchEncoding dict here (not a bare tensor);
            # passing that straight to generate() makes it read dict.shape -> the
            # 'KeyError: shape' -> AttributeError that killed every round. Force the
            # dict form and unpack it (also feeds attention_mask, killing the pad warn).
            enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                          add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)
            enc = {kk: v.to(model.device) for kk, v in enc.items()}
            in_len = enc["input_ids"].shape[1]
            gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=0.95,
                                 num_return_sequences=k, max_new_tokens=24,
                                 pad_token_id=tok.pad_token_id)
            cand = [tok.decode(g[in_len:], skip_special_tokens=True).strip()
                    for g in gen]
            outs.append(cand)
        return outs

    base = load_sft(args.data)
    en_pool = [p for p, _ in base]

    def run_eval(rnd_tag):
        """Frozen eval -> RESULTS.tsv; never fatal."""
        try:
            import subprocess as _sp
            _sp.run([sys.executable, os.path.join(_HERE, "eval_harness.py"),
                     "--model", ckpt, "--round", str(rnd_tag)], timeout=1800)
        except Exception as _e:
            print(f"[eval skipped: {_e}]")

    # ---- SFT-ONLY: pure supervised training on the GOLD corpus, no reward loop.
    # This is the robust path -- it cannot stall (no bootstrapping), and GOLD
    # pairs are verified, so the model directly learns the skill.
    if args.sft_only:
        print(f"SFT-only: {args.epochs} epochs on {len(base)} GOLD pairs -> {ckpt}")
        sft(base, epochs=args.epochs, tag="sftonly")
        model.save_pretrained(ckpt); tok.save_pretrained(ckpt)
        json.dump({"base": args.base, "mode": "sft_only", "epochs": args.epochs,
                   "data": args.data, "pairs": len(base)},
                  open(os.path.join(ckpt, "meta.json"), "w"))
        write_status(round=-1, phase="sft-only-done", pairs=len(base))
        run_eval(-1)
        print(f"SFT-only done -> {ckpt}  (RESULTS.tsv has the frozen eval)")
        return

    if not resumed:
        print(f"SFT warm-start on {len(base)} pairs ...")
        sft(base, epochs=2, tag="sft0")
        model.save_pretrained(ckpt); tok.save_pretrained(ckpt)
        write_status(round=-1, phase="warm-start-done")
        run_eval(-1)                       # measure the warm-start immediately
    else:
        run_eval(start_round - 1)          # measure the checkpoint we resumed

    # adaptive keep-threshold: bootstrapping from a base model that can't yet
    # make homophones needs a LOW bar, ratcheting up as the model improves.
    # (The 894-no-op-round stall was keep_thresh=0.55 with nothing ever kept.)
    eff_thresh = min(args.keep_thresh, 0.42)
    THRESH_FLOOR, THRESH_CEIL = 0.35, args.keep_thresh

    import itertools
    rounds = itertools.count(start_round) if args.continual \
        else range(start_round, args.rounds)
    consec_fail = 0
    for rnd in rounds:
      try:
        batch = random.sample(en_pool, min(256, len(en_pool)))
        new, rewards, samples = [], [], []
        for prompt, cands in zip(batch, sample(batch, args.k)):
            en = prompt.split(":", 1)[-1].strip()
            best, bestr = None, -1.0
            for fr in cands:
                fr = fr.split("\n")[0].strip()
                r = R.reward(en, fr)
                if r > bestr:
                    bestr, best = r, fr
            if best and bestr >= eff_thresh:
                new.append((prompt, best)); rewards.append(bestr)
                if len(samples) < 8:
                    samples.append({"en": en, "fr": best, "reward": round(bestr, 3)})
        mean_r = round(sum(rewards) / len(rewards), 3) if rewards else 0.0
        # anneal the bar: nothing kept -> lower it (unblock); plenty kept -> raise
        if len(new) == 0:
            eff_thresh = round(max(THRESH_FLOOR, eff_thresh - 0.03), 3)
        elif len(new) > 20:
            eff_thresh = round(min(THRESH_CEIL, eff_thresh + 0.02), 3)
        # optional: how does the real LLM judge rate this round's bests?
        if judge and samples:
            for s, l in zip(samples, judge.batch([s["fr"] for s in samples])):
                s["llm_fr"] = round(l, 2)
        print(f"round {rnd}: kept {len(new)}/{len(batch)}, mean reward {mean_r}, "
              f"thresh {eff_thresh}")
        write_status(round=rnd, kept=len(new), batch=len(batch),
                     mean_reward=mean_r, keep_thresh=eff_thresh,
                     base=args.base, samples=samples)
        consec_fail = 0          # this round reached status-write cleanly
        if new:
            sft(new, epochs=1, tag=f"r{rnd}")
            tmp = ckpt + ".tmp"
            model.save_pretrained(tmp); tok.save_pretrained(tmp)     # atomic:
            import shutil                                            # never a
            for f_ in os.listdir(tmp):                               # corrupt
                shutil.move(os.path.join(tmp, f_), os.path.join(ckpt, f_))
            shutil.rmtree(tmp, ignore_errors=True)
            json.dump({"base": args.base, "round": rnd, "data": args.data,
                       "kept": len(new), "mean_reward": mean_r},
                      open(os.path.join(ckpt, "meta.json"), "w"))
            # THE DATA WE WANT: frozen eval every round -> RESULTS.tsv
            import subprocess as _sp
            _sp.run([sys.executable,
                     os.path.join(_HERE, "eval_harness.py"),
                     "--model", ckpt, "--round", str(rnd)], timeout=1800)
      except KeyboardInterrupt:
        break
      except Exception as e:
        import traceback
        tb = traceback.format_exc()
        traceback.print_exc()
        consec_fail += 1
        if "out of memory" in tb.lower():
            import torch as _t
            _t.cuda.empty_cache()
            print("OOM: cleared cache; will retry")
        # Persist the FULL traceback where the supervisor can actually read it
        # (Drive ckpt dir + status.json). str(e) is empty for bare asserts, so
        # the old loop logged nothing and looked 'silently healthy' at round
        # 894 while crashing every round. env=versions catches trl/tfm skew.
        try:
            with open(os.path.join(ckpt, "TRAIN_ERRORS.log"), "a") as ef:
                ef.write(f"\n=== round {rnd} @ {time.strftime('%F %T')} "
                         f"(consec={consec_fail}) ===\n{_env_versions()}\n{tb}")
        except Exception:
            pass
        write_status(round=rnd, error=(str(e) or repr(e) or type(e).__name__),
                     error_type=type(e).__name__, consec_fail=consec_fail,
                     traceback=tb[-2000:], env=_env_versions())
        if not args.continual:           # skip the bad round and keep going
            raise
        # A deterministic crash spins forever without picking up fixes, because a
        # --continual process never restarts itself. After a run of identical
        # failures, EXIT so the supervising notebook loop relaunches us -- that
        # relaunch does `git pull` first, so a pushed fix actually takes effect.
        if consec_fail >= 8:
            print(f"[STUCK] {consec_fail} consecutive round crashes -- exiting so "
                  f"the supervisor loop relaunches with pulled fixes. See "
                  f"{os.path.join(ckpt, 'TRAIN_ERRORS.log')}", flush=True)
            sys.exit(3)
        time.sleep(15)
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    write_status(round=rnd, phase="done", out=args.out)
    print(f"saved self-learned carver -> {args.out}  (status: {status_path})")


if __name__ == "__main__":
    main()
