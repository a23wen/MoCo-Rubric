"""Explicit-endpoint API transport; credentials are read only from the environment."""
import base64
import json
import mimetypes
import os
from pathlib import Path
import urllib.request


def request_text(messages, *, model, api_url, api_key_env, provider="openai",
                 image=None, schema=None, max_tokens=4096, temperature=0.0,
                 top_p=1.0, timeout=180, auth_header=None, auth_prefix=None,
                 disable_thinking=False):
    key = os.environ.get(api_key_env)
    if not key:
        raise ValueError(f"Set {api_key_env}")
    if not api_url or not model:
        raise ValueError("An explicit API URL and model are required")
    encoded = mime = None
    if image is not None:
        path = Path(image)
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}:
            raise ValueError("Unsupported image type")
        encoded = base64.b64encode(path.read_bytes()).decode()
    if provider == "gemini":
        contents = [{"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]
        if encoded:
            contents[0]["parts"].insert(0, {"inlineData": {"mimeType": mime, "data": encoded}})
        payload = {"contents": contents, "generationConfig": {"temperature": temperature,
                   "topP": top_p, "maxOutputTokens": max_tokens}}
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if schema:
            payload["generationConfig"].update(responseMimeType="application/json", responseJsonSchema=schema)
        header = auth_header or "x-goog-api-key"
        prefix = auth_prefix if auth_prefix is not None else ""
    else:
        payload = {"model": model, "messages": [dict(m) for m in messages],
                   "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        if encoded:
            first = next(m for m in payload["messages"] if m["role"] == "user")
            first["content"] = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                                {"type": "text", "text": first["content"]}]
        if schema:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "caption_judgment", "strict": True, "schema": schema}}
        if disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        header = auth_header or "Authorization"
        prefix = auth_prefix if auth_prefix is not None else "Bearer "
    request = urllib.request.Request(api_url, data=json.dumps(payload).encode(),
              headers={"Content-Type": "application/json", header: prefix + key}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    if body.get("error") or body.get("Error"):
        raise ValueError("API returned an error")
    if provider == "gemini":
        result = body["candidates"][0]
        if result.get("finishReason") != "STOP":
            raise ValueError(f"Incomplete Gemini response: {result.get('finishReason')}")
        text = "".join(p.get("text", "") for p in result["content"]["parts"] if not p.get("thought"))
    else:
        result = body["choices"][0]
        if result.get("finish_reason") != "stop":
            raise ValueError(f"Incomplete response: {result.get('finish_reason')}")
        text = result["message"].get("content") or ""
    if not text.strip():
        raise ValueError("Empty API response")
    return text, body
