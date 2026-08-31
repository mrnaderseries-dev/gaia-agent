"""Direct Ollama timing test: plain chat vs structured format."""
import asyncio
import time
import json

import httpx


BASE = "http://localhost:11434"


async def timed_call(payload, label):
    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
    try:
        async with httpx.AsyncClient(base_url=BASE, timeout=180.0) as c:
            r = await c.post("/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
        dt = time.time() - t0
        content = data.get("message", {}).get("content", "")
        print(f"OK in {dt:.1f}s, content[:200]={content[:200]!r}")
        return content
    except Exception as e:
        print(f"FAILED after {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:200]}")
        return None


async def main():
    base = {
        "model": "qwen2.5:3b",
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 512},
    }

    # 1. plain chat (warm-up + baseline)
    await timed_call({**base, "messages": [{"role": "user", "content": "Say hello."}]}, "plain /api/chat")

    # 2. structured format with a JSON schema
    schema = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "step_type": {"type": "string", "enum": ["tool", "llm"]},
                        "tool_name": {"type": ["string", "null"]},
                        "arguments": {"type": "object"},
                        "is_final_answer": {"type": "boolean"},
                    },
                    "required": ["action", "step_type"],
                },
            }
        },
        "required": ["steps"],
    }
    await timed_call(
        {
            **base,
            "messages": [
                {"role": "user", "content": "Return a JSON plan with one step to search the web for population of France."}
            ],
            "format": schema,
        },
        "structured /api/chat (format-schema)",
    )


if __name__ == "__main__":
    asyncio.run(main())
