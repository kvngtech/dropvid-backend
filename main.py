import os
import re
import time
import secrets
import asyncio
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask
import yt_dlp
import imageio_ffmpeg

app = FastAPI(title="DropVid API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()


def normalize_netscape_cookies(raw: str) -> str:
    """
    Rebuilds a Netscape-format cookies file field-by-field so it survives
    being pasted through a browser textarea, which often collapses tabs
    into spaces. Each real cookie line has exactly 7 whitespace-separated
    fields; comments/blank lines are passed through untouched.
    """
    lines_out = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            lines_out.append(line if stripped else "")
            continue
        fields = stripped.split()
        if len(fields) >= 7:
            # Cookie values can't legally contain whitespace, so this is safe.
            fields = fields[:6] + [" ".join(fields[6:])]
            lines_out.append("\t".join(fields))
        else:
            lines_out.append(line)  # leave anything unexpected as-is
    return "\n".join(lines_out) + "\n"


# Render mounts "Secret Files" at /etc/secrets/<filename>. As a fallback
# (some browsers mangle tabs when pasting into that box), we also accept
# the raw cookies.txt contents pasted into a plain Environment Variable
# called YOUTUBE_COOKIES_CONTENT. Either way, we rewrite the content into
# a fresh, correctly-formatted temp file before handing it to yt-dlp.
COOKIES_FILE = None

_secret_candidates = [
    os.environ.get("COOKIES_FILE", ""),
    "/etc/secrets/cookies.txt",
    os.path.join(os.path.dirname(__file__), "cookies.txt"),
]
_secret_path = next((p for p in _secret_candidates if p and os.path.isfile(p)), None)

_raw_cookies = None
if _secret_path:
    with open(_secret_path, "r") as f:
        _raw_cookies = f.read()
elif os.environ.get("YOUTUBE_COOKIES_CONTENT", "").strip():
    _raw_cookies = os.environ["YOUTUBE_COOKIES_CONTENT"]

if _raw_cookies:
    _normalized = normalize_netscape_cookies(_raw_cookies)
    _tmp_cookie_path = os.path.join(tempfile.gettempdir(), "dropvid_cookies.txt")
    with open(_tmp_cookie_path, "w") as f:
        f.write(_normalized)
    COOKIES_FILE = _tmp_cookie_path


def base_ydl_opts() -> dict:
    opts = {"quiet": True, "no_warnings": True}
    if COOKIES_FILE:
        opts["cookiefile"] = COOKIES_FILE
    return opts


# ──────────────────────────────────────────────────────────────
# Limits. Render's free instance has ~512MB RAM, so we cap how many
# yt-dlp lookups / ffmpeg processes can run at the same time.
# ──────────────────────────────────────────────────────────────
CHUNK = 128 * 1024
EXTRACT_SLOTS = threading.BoundedSemaphore(4)
FFMPEG_SLOTS = threading.BoundedSemaphore(3)

# Short-lived "download tickets". /prepare resolves the stream once and
# hands the browser a token; the browser then downloads /file/<token>
# natively. In-memory is fine because the app runs as a single process
# (see Procfile). If you ever run several workers, move this to Redis.
PLAN_TTL = 15 * 60
PLANS: dict = {}
PLANS_LOCK = threading.Lock()


class URLRequest(BaseModel):
    urls: List[str]


class DownloadRequest(BaseModel):
    url: str
    quality: str = "worst"  # "1080", "720", "480", "360", "240", "worst" (Data Saver), "audio"


def get_video_info(url: str):
    ydl_opts = {
        **base_ydl_opts(),
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        formats = info.get("formats", [])
        duration = info.get("duration", 0) or 0

        def fmt_size(f):
            return f.get("filesize") or f.get("filesize_approx") or 0

        audio_formats = [f for f in formats if f.get("vcodec") == "none" and f.get("acodec") != "none"]
        audio_size = max((fmt_size(f) for f in audio_formats), default=0)
        if not audio_size and duration:
            audio_size = int(duration * 128 * 1000 / 8)

        BUCKETS = [1080, 720, 480, 360, 240]
        bucket_sizes = {}
        for b in BUCKETS:
            candidates = [f for f in formats if f.get("height") and b - 40 <= f.get("height") <= b + 40]
            if not candidates:
                continue
            best = max(candidates, key=fmt_size)
            size = fmt_size(best)
            if size and best.get("vcodec") != "none" and best.get("acodec") == "none":
                size += audio_size
            bucket_sizes[b] = size

        # "Data Saver" = the smallest real stream available, mirroring what
        # sites like fdown serve by default (the platform's own low-bitrate
        # SD encode), rather than the best stream within a height cap.
        muxed = [f for f in formats if f.get("vcodec") not in (None, "none") and f.get("acodec") not in (None, "none") and fmt_size(f) > 0]
        if muxed:
            smallest = min(muxed, key=fmt_size)
            data_saver_size = fmt_size(smallest)
        else:
            video_only = [f for f in formats if f.get("vcodec") not in (None, "none") and f.get("acodec") in (None, "none") and fmt_size(f) > 0]
            if video_only:
                smallest = min(video_only, key=fmt_size)
                data_saver_size = fmt_size(smallest) + audio_size
            else:
                data_saver_size = 0

        qualities = []
        data_saver_mb = round(data_saver_size / 1_000_000, 1) if data_saver_size else None
        qualities.append({"value": "worst", "label": "Data Saver", "size_mb": data_saver_mb})

        for b in sorted(bucket_sizes.keys(), reverse=True):
            size_mb = round(bucket_sizes[b] / 1_000_000, 1) if bucket_sizes[b] else None
            qualities.append({"value": str(b), "label": f"{b}p", "size_mb": size_mb})

        audio_size_mb = round(audio_size / 1_000_000, 1) if audio_size else None
        qualities.append({"value": "audio", "label": "Audio (MP3)", "size_mb": audio_size_mb})

        return {
            "title": info.get("title", "Unknown Title"),
            "thumbnail": info.get("thumbnail", ""),
            "duration": duration,
            "platform": info.get("extractor_key", "Unknown"),
            "qualities": qualities,
        }


def get_video_info_limited(url: str):
    with EXTRACT_SLOTS:
        return get_video_info(url)


# ──────────────────────────────────────────────────────────────
# Resolving what to download (no bytes are downloaded here)
# ──────────────────────────────────────────────────────────────
def validate_quality(quality: str) -> str:
    if quality in ("worst", "audio") or (quality.isdigit() and 100 <= int(quality) <= 4320):
        return quality
    raise HTTPException(status_code=400, detail="Unknown quality option")


def build_format_string(quality: str) -> str:
    if quality == "audio":
        return "bestaudio/best"
    if quality == "worst":
        # Smallest single file that already has video + audio together.
        return "worst[vcodec!=none][acodec!=none]/worst"
    return f"bv*[height<={quality}]+ba/b[height<={quality}]/b"


def clean_headers(headers) -> dict:
    out = dict(headers or {})
    for key in list(out):
        if key.lower() in ("accept-encoding", "host", "content-length", "range"):
            del out[key]
    return out


MIME_BY_EXT = {
    "mp4": "video/mp4",
    "webm": "video/webm",
    "mov": "video/quicktime",
    "m4a": "audio/mp4",
    "mp3": "audio/mpeg",
}


def resolve_plan(url: str, quality: str) -> dict:
    opts = {
        **base_ydl_opts(),
        "skip_download": True,
        "noplaylist": True,
        "format": build_format_string(quality),
    }
    if quality not in ("audio", "worst"):
        # Highest resolution under the cap, preferring mp4/m4a so the
        # streams can be joined without re-encoding.
        opts["format_sort"] = ["res", "ext:mp4:m4a"]

    with EXTRACT_SLOTS:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)

    if info and info.get("entries"):
        first = next(iter(info["entries"]), None)
        if first:
            info = first
    if not info:
        raise RuntimeError("Could not read this video")

    chosen = info.get("requested_formats") or [info]
    sources = []
    for f in chosen:
        if not f.get("url"):
            raise RuntimeError("No downloadable stream found for this video")
        sources.append({
            "url": f["url"],
            "headers": clean_headers(f.get("http_headers") or info.get("http_headers")),
            "protocol": f.get("protocol") or "",
            "acodec": f.get("acodec") or "",
            "ext": f.get("ext") or "",
            "size": f.get("filesize") or f.get("filesize_approx") or 0,
        })

    title = re.sub(r"[^\w\- ]+", "", info.get("title") or "video", flags=re.ASCII).strip()
    title = re.sub(r"\s+", "_", title)[:60] or "video"

    if quality == "audio":
        mode, ext, inputs = "mp3", "mp3", [sources[0]]
    elif len(sources) == 1 and sources[0]["protocol"] in ("http", "https"):
        # One ready-made file on a normal web server: just pass it through.
        mode, ext, inputs = "direct", (sources[0]["ext"] or "mp4"), sources
    else:
        # Separate video+audio (or an HLS/DASH stream): join on the fly.
        mode, ext, inputs = "mux", "mp4", sources

    return {
        "mode": mode,
        "inputs": inputs,
        "ext": ext,
        "mime": MIME_BY_EXT.get(ext, "application/octet-stream"),
        "filename": f"dropvid_{title}.{ext}",
        "size": sources[0]["size"] if mode == "direct" else None,
    }


def probe_direct(src: dict):
    """Cheap check that the source server will actually serve the file, so
    problems show up on the page instead of as a silent failed download."""
    req = urllib.request.Request(src["url"], headers={**src["headers"], "Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            r.read(1)
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"The source server refused the download (HTTP {e.code}). Try another quality.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not reach the source server: {e}")


def make_plan(url: str, quality: str) -> dict:
    quality = validate_quality(quality)
    try:
        plan = resolve_plan(url.strip(), quality)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if plan["mode"] == "direct":
        probe_direct(plan["inputs"][0])
    return plan


def store_plan(plan: dict) -> str:
    token = secrets.token_urlsafe(16)
    now = time.time()
    with PLANS_LOCK:
        for old in [t for t, (ts, _) in PLANS.items() if now - ts > PLAN_TTL]:
            del PLANS[old]
        PLANS[token] = (now, plan)
    return token


def get_plan(token: str):
    with PLANS_LOCK:
        item = PLANS.get(token)
    if not item or time.time() - item[0] > PLAN_TTL:
        return None
    return item[1]


# ──────────────────────────────────────────────────────────────
# Streaming the bytes to the browser
# ──────────────────────────────────────────────────────────────
class ByteStream:
    """Iterable of byte chunks that always cleans up after itself, even if
    the client disconnects before the first chunk is read."""

    def __init__(self, read_chunk, on_close, first: bytes = b""):
        self._read_chunk = read_chunk
        self._on_close = on_close
        self._first = first
        self._closed = False
        self._lock = threading.Lock()
        self.length = None

    def __iter__(self):
        try:
            if self._first:
                yield self._first
            while True:
                chunk = self._read_chunk()
                if not chunk:
                    break
                yield chunk
        finally:
            self.close()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._on_close()
        except Exception:
            pass


def build_ffmpeg_cmd(plan: dict) -> list:
    cmd = [FFMPEG_PATH, "-hide_banner", "-loglevel", "error", "-nostdin"]
    for src in plan["inputs"]:
        header_block = "".join(f"{k}: {v}\r\n" for k, v in src["headers"].items())
        if header_block:
            cmd += ["-headers", header_block]
        cmd += ["-i", src["url"]]

    if plan["mode"] == "mp3":
        return cmd + ["-vn", "-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3", "pipe:1"]

    inputs = plan["inputs"]
    if len(inputs) == 2:
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]
        audio_src = inputs[1]
    else:
        cmd += ["-map", "0:v:0?", "-map", "0:a:0?"]
        audio_src = inputs[0]

    cmd += ["-c:v", "copy"]
    if audio_src["acodec"].startswith(("mp4a", "aac")):
        cmd += ["-c:a", "copy"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "128k"]

    # Fragmented MP4 can be written to a pipe, so the download starts
    # immediately instead of waiting for the whole file to be built.
    cmd += ["-movflags", "frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", "pipe:1"]
    return cmd


def open_direct(src: dict) -> ByteStream:
    req = urllib.request.Request(src["url"], headers=src["headers"])
    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"The source server refused the download (HTTP {e.code}). Please try again.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not reach the source server: {e}")
    stream = ByteStream(lambda: resp.read(CHUNK), resp.close)
    stream.length = resp.headers.get("Content-Length")
    return stream


def open_ffmpeg(plan: dict) -> ByteStream:
    if not FFMPEG_SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="The server is busy right now. Please try again in a few seconds.")

    errlog = tempfile.TemporaryFile()
    proc = None

    def release():
        try:
            if proc is not None:
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=5)
        except Exception:
            pass
        try:
            errlog.close()
        except Exception:
            pass
        FFMPEG_SLOTS.release()

    try:
        proc = subprocess.Popen(
            build_ffmpeg_cmd(plan),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=errlog,
            bufsize=0,
        )
        first = proc.stdout.read(CHUNK)
    except Exception as e:
        release()
        raise HTTPException(status_code=500, detail=f"Could not start the converter: {e}")

    if not first:
        errlog.seek(0)
        tail = errlog.read()[-400:].decode("utf-8", "replace").strip()
        release()
        print(f"[ffmpeg] produced no output: {tail}", flush=True)
        raise HTTPException(status_code=502, detail="Could not read this video's stream. Try another quality.")

    return ByteStream(lambda: proc.stdout.read(CHUNK), release, first=first)


def open_stream(plan: dict) -> StreamingResponse:
    headers = {
        "Content-Disposition": f'attachment; filename="{plan["filename"]}"',
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",
    }
    if plan["mode"] == "direct":
        stream = open_direct(plan["inputs"][0])
        if stream.length:
            headers["Content-Length"] = str(stream.length)
    else:
        stream = open_ffmpeg(plan)

    return StreamingResponse(
        iter(stream),
        media_type=plan["mime"],
        headers=headers,
        background=BackgroundTask(stream.close),
    )


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "DropVid API is running \U0001f680",
        "cookies_loaded": bool(COOKIES_FILE),
    }


@app.post("/info")
async def fetch_info(request: URLRequest):
    if len(request.urls) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 URLs allowed")

    async def one(url: str):
        try:
            info = await asyncio.to_thread(get_video_info_limited, url)
            return {"url": url, "success": True, **info}
        except Exception as e:
            return {"url": url, "success": False, "error": str(e)}

    urls = [u.strip() for u in request.urls if u.strip()]
    results = await asyncio.gather(*[one(u) for u in urls])
    return {"results": list(results)}


@app.post("/prepare")
def prepare_download(request: DownloadRequest):
    """Step 1: work out what to download and hand back a short-lived ticket."""
    plan = make_plan(request.url, request.quality)
    token = store_plan(plan)
    return {"token": token, "filename": plan["filename"], "size": plan["size"]}


@app.get("/file/{token}")
def download_file(token: str):
    """Step 2: the browser downloads this URL natively and streams the bytes."""
    plan = get_plan(token)
    if not plan:
        raise HTTPException(status_code=404, detail="This download link expired. Please click Download again.")
    return open_stream(plan)


@app.post("/download")
def download_video(request: DownloadRequest):
    """Kept so an older cached copy of the website keeps working."""
    plan = make_plan(request.url, request.quality)
    return open_stream(plan)
