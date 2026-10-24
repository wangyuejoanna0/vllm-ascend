# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""Send one deterministic request for MoE dump A/B comparison.

Pair with ``VLLM_ASCEND_MOE_DEBUG_DUMP``: the instrumentation records the first
fused-MoE call of the first real request, so both the MegaMoe and the CAM run
must receive byte-identical input. This sends a single greedy request, waits
for the server to be ready first, and optionally saves the response.

Usage:
    python tools/moe_debug/send_moe_debug_request.py --port 8000
    python tools/moe_debug/send_moe_debug_request.py --port 8000 --save /tmp/cam_response.json
    python tools/moe_debug/send_moe_debug_request.py --port 8000 --completions

Keep these scripts in a subdirectory: Python puts the script's own directory on
sys.path, and a sibling ``bisect`` package under tools/ shadows the stdlib
module, breaking any script that imports ``random``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# Kept fixed and short: a single greedy prefill step makes the recorded MoE
# inputs deterministic and identical across the two backends.
DEFAULT_PROMPT = "1+1 等于几？只回答数字。"
DEFAULT_SYSTEM = "You are a helpful assistant."


# Deployments behind an HTTP proxy (common on NPU hosts) would otherwise send
# localhost requests to the proxy, which hangs until the socket times out. This
# mirrors `curl --noproxy '*'` and keeps the request on the loopback interface.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def wait_for_server(base: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    attempt = 0
    last_error = "no attempt yet"
    while time.monotonic() < deadline:
        attempt += 1
        try:
            with _OPENER.open(f"{base}/health", timeout=10) as response:
                if response.status == 200:
                    print(f"server is ready (after {attempt} probe(s))")
                    return
                last_error = f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = str(getattr(exc, "reason", exc))
        print(f"  waiting for {base}/health (attempt {attempt}): {last_error}", flush=True)
        time.sleep(5)
    raise SystemExit(f"server at {base} did not become healthy within {timeout:.0f}s; last probe: {last_error}")


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _OPENER.open(request, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:  # surface the server's message
        raise SystemExit(f"HTTP {exc.code} from {url}: {exc.read().decode(errors='replace')}") from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise SystemExit(f"cannot reach {url}: {reason}\nIs the server listening on that host/port?") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="glm-5")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="system message; pass '' to omit it")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for server startup")
    parser.add_argument("--completions", action="store_true", help="use /v1/completions with a raw prompt")
    parser.add_argument("--save", default=None, help="write the raw response JSON here")
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    wait_for_server(base, args.timeout)

    if args.completions:
        url = f"{base}/v1/completions"
        payload = {
            "model": args.model,
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 1024,
        }
    else:
        url = f"{base}/v1/chat/completions"
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 1024,
        }

    print(f"POST {url}", flush=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    response = post_json(url, payload)

    if args.save:
        with open(args.save, "w") as handle:
            json.dump(response, handle, ensure_ascii=False, indent=2)
        print(f"response written to {args.save}")

    choice = response["choices"][0]
    if args.completions:
        text = choice.get("text", "")
        prompt_tokens = response["usage"]["prompt_tokens"]
    else:
        text = choice["message"]["content"]
        prompt_tokens = response["usage"]["prompt_tokens"]

    print(f"\nprompt_tokens = {prompt_tokens}   completion_tokens = {response['usage']['completion_tokens']}")
    print(f"finish_reason = {choice.get('finish_reason')}")
    print(f"output = {text!r}")
    print("\nBoth runs must report the same prompt_tokens; a mismatch means the")
    print("two servers did not receive identical input and the dumps are not comparable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
