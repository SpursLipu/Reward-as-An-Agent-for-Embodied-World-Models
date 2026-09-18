"""Localhost-only, model-blinded human video review using a frozen manifest."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import parse_qs, urlsplit

from scripts.human.human_alignment import RUBRIC_VERSION, read_jsonl, unit_score, validate_manifest


HTML = r"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>视频人工复核</title>
<style>
body{font:16px/1.6 system-ui,sans-serif;max-width:1050px;margin:24px auto;padding:0 18px;color:#142333;background:#f6f8fa}
section{background:white;border:1px solid #d3dbe3;border-radius:10px;padding:18px;margin:16px 0}
video{display:block;width:100%;max-height:530px;background:#111;margin:12px 0}label{display:block;margin:12px 0}
input,select,textarea,button{font:inherit;padding:7px}input[type=number]{width:100px}textarea{width:96%;min-height:70px}
button{cursor:pointer;background:#175eaf;color:white;border:0;border-radius:5px;padding:9px 20px}
button:disabled{background:#8a99ab;cursor:wait}#message{white-space:pre-wrap;color:#9b2222}#prompt{white-space:pre-wrap}
.row{display:flex;flex-wrap:wrap;gap:25px}.muted{color:#546375}details{margin:10px 0}h1{font-size:25px}
</style>
<h1>视频人工复核</h1>
<p>依据任务和视频给出自己的判断。界面隐藏所有模型评分与解释；请勿同时打开旧评分报告。</p>
<section id="setup"><label>标注者代号 <input id="rater" placeholder="例如 reviewer_01" maxlength="64" autocomplete="off"></label>
<label><input type="checkbox" id="human"> 我是人工观看后填写，答案不是由 AI 代写。</label>
<label><input type="checkbox" id="independent"> 我未看过本批视频的模型分数、解释或其他人的评分，能独立判断。</label>
<button id="begin">开始 / 继续标注</button></section>
<section id="work" hidden><div id="progress" class="muted"></div><h2>任务要求</h2><p id="prompt"></p>
<video id="video" controls preload="metadata" playsinline></video>
<p class="muted">可暂停、拖动和重播。看到遮挡、正常夹持或局部运动模糊时，不要仅凭二维重叠断言穿模。</p>
<details><summary>评分参考（同一批统一使用）</summary><p id="rubric"></p><ul id="anchors"></ul><p id="evidence"></p></details>
<form id="form"><div class="row">
<label>整体 reward（0–1）<br><input id="score" type="number" min="0" max="1" step="0.01" required placeholder="自行判断"></label>
<label>任务完成度<br><select id="task" required><option value="">请选择</option><option value="0">0：未完成</option><option value="0.25">0.25：少量进展</option><option value="0.5">0.5：部分完成</option><option value="0.75">0.75：大部分完成</option><option value="1">1：完整完成</option><option value="unknown">无法观察</option></select></label>
<label>物理问题严重程度<br><select id="physics" required><option value="">请选择</option><option value="none">未见明显问题</option><option value="minor">轻微：局部外观瑕疵</option><option value="moderate">中等：明显影响执行</option><option value="severe">严重：物理不可能使动作失效</option><option value="uncertain">证据不足 / 不确定</option></select></label>
<label>判断置信度<br><select id="confidence" required><option value="">请选择</option><option value="low">低</option><option value="medium">中</option><option value="high">高</option></select></label></div>
<label>具体证据 / 缺陷 / 不确定处（尽量注明时间，如 2.1–2.8 秒）<br><textarea id="notes" required minlength="5" maxlength="4000"></textarea></label>
<label><input id="watched" type="checkbox" required> 我已完整观看这段视频，并按实际可见证据作判断。</label>
<button id="save" type="submit">保存并看下一个</button> <button id="skip" type="button">暂时跳过</button></form></section>
<p id="message" role="status"></p>
<script>
const csrf=__CSRF__, $=id=>document.getElementById(id);let items=[],current=null,rater='',independent=false,completed=0,total=0;
async function request(url, options={}){const response=await fetch(url,options);const body=await response.json();if(!response.ok)throw Error(body.error||response.status);return body}
function next(){if(!items.length){$('work').hidden=true;$('message').textContent='本次队列已完成。评分已保存在服务端的 labels.jsonl；需要独立标注者复核后再计算人与模型的一致性。';return}current=items.shift();$('form').reset();$('prompt').textContent=current.prompt;$('video').src=current.video_url;$('progress').textContent=`已保存 ${completed} / ${total}；当前视频 ${current.review_id.slice(0,8)}`;$('work').hidden=false;$('message').textContent='';}
$('begin').onclick=async()=>{try{rater=$('rater').value.trim();if(!/^[A-Za-z0-9_.-]{1,64}$/.test(rater))throw Error('代号仅使用 1–64 位英文字母、数字、下划线、点或短横线。');if(!$('human').checked)throw Error('请确认这是人工判断。');independent=$('independent').checked;const data=await request('/api/items?rater='+encodeURIComponent(rater));items=data.items;completed=data.completed;total=data.total;$('rubric').textContent=data.rubric.overall;$('evidence').textContent=data.rubric.evidence;$('anchors').replaceChildren(...data.rubric.anchors.map(a=>{const li=document.createElement('li');li.textContent=a.score+'：'+a.text;return li}));$('setup').hidden=true;next()}catch(e){$('message').textContent=e.message}};
$('form').onsubmit=async e=>{e.preventDefault();$('save').disabled=true;try{await request('/api/labels',{method:'POST',headers:{'Content-Type':'application/json','X-Review-CSRF':csrf},body:JSON.stringify({review_id:current.review_id,rater_id:rater,label_source:'human',independent,score:Number($('score').value),task_completion:$('task').value==='unknown'?null:Number($('task').value),physics_severity:$('physics').value,confidence:$('confidence').value,notes:$('notes').value,watched_full_video:$('watched').checked})});completed++;next()}catch(e){$('message').textContent=e.message}finally{$('save').disabled=false}};
$('skip').onclick=()=>{items.push(current);next()};
</script></html>"""


class ReviewStore:
    def __init__(self, manifest: dict, labels_path: Path, video_root: Path, split: str = "development"):
        self.manifest = validate_manifest(manifest)
        self.labels_path = labels_path
        self.video_root = video_root.resolve(strict=True)
        if not self.video_root.is_dir():
            raise ValueError("video-root must be an existing directory")
        if split not in ("development", "heldout", "all"):
            raise ValueError("Invalid review split")
        self.samples = {row["review_id"]: row for row in manifest["samples"] if split == "all" or row["split"] == split}
        self.csrf = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        if labels_path.exists():
            # Reject mixed-manifest files before accepting new annotations.
            for row in read_jsonl(labels_path):
                if row.get("manifest_sha256") != manifest["manifest_sha256"]:
                    raise ValueError("Existing labels belong to a different frozen manifest")

    def video_path(self, review_id: str) -> Path:
        row = self.samples.get(review_id)
        if row is None:
            raise KeyError("Unknown video")
        path = Path(row["video_path"]).resolve(strict=True)
        try:
            path.relative_to(self.video_root)
        except ValueError:
            raise ValueError("Manifest video lies outside the explicitly allowed video-root") from None
        if path.suffix.lower() != ".mp4" or not path.is_file():
            raise ValueError("Not an allowed MP4 video")
        return path

    def queue(self, rater: str) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", rater):
            raise ValueError("Invalid rater ID")
        with self.lock:
            prior = read_jsonl(self.labels_path) if self.labels_path.exists() else []
        completed = {row["sample_id"] for row in prior if row.get("rater_id") == rater}
        ordered = sorted(self.samples.values(), key=lambda row: hashlib.sha256((rater + row["review_id"]).encode()).hexdigest())
        # Explicit allow-list: never serialize full result records into the UI.
        items = [{"review_id": row["review_id"], "prompt": row["prompt"], "video_url": "/video/" + row["review_id"]}
                 for row in ordered if row["sample_id"] not in completed]
        return {"items": items, "total": len(ordered), "completed": len(ordered) - len(items), "rubric": self.manifest["rubric"]}

    def save(self, payload: dict) -> dict:
        row = self.samples.get(payload.get("review_id"))
        if row is None:
            raise ValueError("Unknown review ID")
        rater = payload.get("rater_id")
        if not isinstance(rater, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", rater):
            raise ValueError("Invalid rater ID")
        if payload.get("label_source") != "human" or payload.get("watched_full_video") is not True:
            raise ValueError("An actual human must watch the complete video and explicitly declare it")
        if not isinstance(payload.get("independent"), bool):
            raise ValueError("Missing independence declaration")
        score = unit_score(payload.get("score"))
        completion = payload.get("task_completion")
        if completion is not None:
            completion = unit_score(completion)
            if completion not in (0, .25, .5, .75, 1):
                raise ValueError("Invalid task completion category")
        if payload.get("confidence") not in ("low", "medium", "high"):
            raise ValueError("Invalid confidence")
        if payload.get("physics_severity") not in ("none", "minor", "moderate", "severe", "uncertain"):
            raise ValueError("Invalid physics severity")
        notes = payload.get("notes")
        if not isinstance(notes, str) or not 5 <= len(notes.strip()) <= 4000:
            raise ValueError("Evidence notes must contain 5–4000 characters")
        # Validate that an allowed video exists; labels cannot target arbitrary paths.
        self.video_path(row["review_id"])
        label = {
            "sample_id": row["sample_id"], "review_id": row["review_id"], "rater_id": rater,
            "manifest_sha256": self.manifest["manifest_sha256"], "rubric_version": RUBRIC_VERSION,
            "label_source": "human", "blinded": True, "independent": payload["independent"],
            "score": score, "task_completion": completion,
            "physics_severity": payload["physics_severity"], "confidence": payload["confidence"],
            "notes": notes.strip(), "watched_full_video": True,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.lock:
            # Append is crash-resilient and preserves corrections. Metrics select
            # the latest record for each (sample, rater) pair.
            with self.labels_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(label, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return {"saved": True, "review_id": row["review_id"]}


def make_handler(store: ReviewStore):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            # No task text, source paths, annotations, or rater identifiers in access logs.
            pass

        def permitted_host(self):
            port = self.server.server_address[1]
            return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def send_bytes(self, body: bytes, content_type: str, status: int = 200, headers: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def json_response(self, value, status=200):
            self.send_bytes(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8", status)

        def do_GET(self):
            if not self.permitted_host():
                self.json_response({"error": "Localhost Host header required"}, 403)
                return
            route = urlsplit(self.path)
            try:
                if route.path == "/":
                    body = HTML.replace("__CSRF__", json.dumps(store.csrf)).encode()
                    self.send_bytes(body, "text/html; charset=utf-8", headers={"Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"})
                elif route.path == "/api/items":
                    rater = parse_qs(route.query).get("rater", [""])[0]
                    self.json_response(store.queue(rater))
                elif re.fullmatch(r"/video/[a-f0-9]{24}", route.path):
                    self.serve_video(route.path.rsplit("/", 1)[1])
                else:
                    self.json_response({"error": "Not found"}, 404)
            except KeyError:
                self.json_response({"error": "Not found"}, 404)
            except (ValueError, FileNotFoundError):
                self.json_response({"error": "Unavailable allowed video or invalid request"}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def serve_video(self, review_id):
            path = store.video_path(review_id)
            with path.open("rb") as handle:
                size = os.fstat(handle.fileno()).st_size
                start, end, status = 0, size - 1, 200
                requested = self.headers.get("Range")
                if requested:
                    match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested)
                    if not match or not any(match.groups()):
                        self.send_bytes(b"", "video/mp4", 416, {"Content-Range": f"bytes */{size}"})
                        return
                    first, last = match.groups()
                    if first:
                        start = int(first)
                        end = min(int(last), size - 1) if last else size - 1
                    else:
                        start = max(0, size - int(last))
                    if start > end or start >= size:
                        self.send_bytes(b"", "video/mp4", 416, {"Content-Range": f"bytes */{size}"})
                        return
                    status = 206
                self.send_response(status)
                self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(max(0, end - start + 1)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    block = handle.read(min(65536, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def do_POST(self):
            if not self.permitted_host() or self.headers.get("X-Review-CSRF") != store.csrf:
                self.close_connection = True
                self.json_response({"error": "Invalid local review request"}, 403)
                return
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.close_connection = True
                self.json_response({"error": "Cross-origin writes are not accepted"}, 403)
                return
            if self.path != "/api/labels":
                self.close_connection = True
                self.json_response({"error": "Not found"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json" or not 0 < length <= 32768:
                    raise ValueError("Expected a bounded JSON annotation")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError("Expected JSON object")
                self.json_response(store.save(value))
            except (ValueError, UnicodeError, FileNotFoundError) as exc:
                self.close_connection = True
                self.json_response({"error": str(exc)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True, help="Only manifest MP4 files beneath this directory can be served")
    parser.add_argument("--split", choices=("development", "heldout", "all"), default="development")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    store = ReviewStore(json.loads(args.manifest.read_text()), args.labels, args.video_root, args.split)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    print(f"Human review: http://127.0.0.1:{server.server_address[1]} — {len(store.samples)} {args.split} videos", flush=True)
    print("Model scores are hidden. This tool collects real human declarations; it cannot authenticate a person's independence.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
