#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import threading
import time
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8091/v1/chat/completions")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output")
    parser.add_argument("--same-prompt", action="store_true")
    parser.add_argument("--pin-slots", action="store_true")
    parser.add_argument("--slot-offset", type=int, default=0)
    parser.add_argument("--raw-completion", action="store_true")
    parser.add_argument("--target-tps", type=float, default=10.0)
    args = parser.parse_args()

    start = threading.Event()

    def request_one(index: int) -> dict:
        prompt_index = 0 if args.same_prompt else index
        if args.raw_completion:
            payload = {
                "prompt": (
                    "Question: Is continuous batching working correctly? "
                    f"Test {prompt_index}. Answer:"
                ),
                "n_predict": args.tokens,
                "temperature": 0,
                "seed": 1,
                "stream": False,
            }
        else:
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"用一句简短中文回答：连续批处理测试 {prompt_index} 是否正常？",
                    }
                ],
                "max_tokens": args.tokens,
                "temperature": 0,
                "seed": 1,
                "stream": False,
            }
        if args.pin_slots:
            payload["id_slot"] = args.slot_offset + index
            payload["cache_prompt"] = True
        body = json.dumps(payload).encode()
        url = args.url
        if args.raw_completion and url.endswith("/v1/chat/completions"):
            url = url[: -len("/v1/chat/completions")] + "/completion"
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        start.wait()
        begin = time.monotonic()
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
        end = time.monotonic()
        choice = result.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = result.get("content", message.get("content", ""))
        reasoning = message.get("reasoning_content", "")
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = result.get(
                "tokens_predicted", result.get("timings", {}).get("predicted_n", 0)
            )
        return {
            "index": index,
            "begin": begin,
            "end": end,
            "seconds": end - begin,
            "completion_tokens": int(completion_tokens),
            "content": content,
            "reasoning_content": reasoning,
            "finish_reason": choice.get("finish_reason"),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(request_one, i) for i in range(args.parallel)]
        release = time.monotonic()
        start.set()
        results = [future.result() for future in futures]

    finish = max(item["end"] for item in results)
    begin = min(item["begin"] for item in results)
    total_tokens = sum(item["completion_tokens"] for item in results)
    wall_seconds = finish - begin
    def semantic_output(item: dict) -> bool:
        text = (item["content"] + item["reasoning_content"]).strip()
        if len(text) < 4:
            return False
        replacement_count = text.count("?") + text.count("\ufffd")
        return replacement_count * 2 < len(text) and len(set(text)) >= 3

    semantic_outputs = sum(semantic_output(item) for item in results)
    summary = {
        "parallel": args.parallel,
        "pin_slots": args.pin_slots,
        "slot_offset": args.slot_offset,
        "raw_completion": args.raw_completion,
        "requested_tokens_each": args.tokens,
        "total_completion_tokens": total_tokens,
        "wall_seconds": wall_seconds,
        "aggregate_tps": total_tokens / wall_seconds if wall_seconds else 0.0,
        "target_tps": args.target_tps,
        "dispatch_skew_ms": (begin - release) * 1000.0,
        "nonempty_outputs": sum(
            bool(item["content"].strip() or item["reasoning_content"].strip())
            for item in results
        ),
        "semantic_outputs": semantic_outputs,
        "results": results,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    if summary["nonempty_outputs"] != args.parallel or semantic_outputs != args.parallel:
        return 2
    return 0 if summary["aggregate_tps"] >= args.target_tps else 3


if __name__ == "__main__":
    raise SystemExit(main())
