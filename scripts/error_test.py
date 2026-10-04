"""Error-handling tests: unsupported, corrupt, empty, missing modalities."""
import io
import json
import os
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:" + os.environ.get("TL_PORT", "8000")


def call(path, method="GET", payload=None, raw=None, ctype=None):
    data = raw if raw is not None else (json.dumps(payload).encode() if payload is not None else None)
    headers = {}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if ctype:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(BASE + path, data=data, headers=headers, method=method)
    try:
        r = urllib.request.urlopen(req)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def multipart(filename, content):
    b = "----tl"
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + content + f"\r\n--{b}--\r\n".encode()
    return body, f"multipart/form-data; boundary={b}"


# create investigation
_, inv = call("/api/investigations", "POST", {"name": "Error Tests"})
inv_id = inv["id"]

# 1. unsupported file type
body, ct = multipart("malware.exe", b"MZ\x90\x00binary")
s, r = call(f"/api/investigations/{inv_id}/upload", "POST", raw=body, ctype=ct)
print("unsupported ->", s, r.get("detail"))

# 2. empty file
body, ct = multipart("empty.jpg", b"")
s, r = call(f"/api/investigations/{inv_id}/upload", "POST", raw=body, ctype=ct)
print("empty ->", s, r.get("detail"))

# 3. corrupt image
body, ct = multipart("broken.jpg", b"not really a jpeg")
s, r = call(f"/api/investigations/{inv_id}/upload", "POST", raw=body, ctype=ct)
print("corrupt upload accepted ->", s)

# 4. analyze with only corrupt image -> should not crash
s, r = call(f"/api/investigations/{inv_id}/analyze", "POST", {})
print("analyze corrupt ->", s, "assessment:", r.get("uncertainty", {}).get("final_assessment"),
      "status:", r.get("fusion", {}).get("engine"))
print("  errors:", r.get("errors"))
ind = r.get("individual", {})
for m, a in ind.items():
    print("  ", m, "quality:", a.get("quality"), "label:", a.get("label"))

# 5. analyze nonexistent investigation
s, r = call("/api/investigations/does-not-exist/analyze", "POST", {})
print("analyze missing inv ->", s, r.get("detail"))

# 6. analyze investigation with no evidence
_, inv2 = call("/api/investigations", "POST", {"name": "Empty Inv"})
s, r = call(f"/api/investigations/{inv2['id']}/analyze", "POST", {})
print("analyze no evidence ->", s, r.get("detail"))

# 7. get graph for missing
s, r = call("/api/investigations/nope/graph")
print("graph missing ->", s, r.get("detail"))

# 8. name validation
s, r = call("/api/investigations", "POST", {"name": ""})
print("empty name ->", s)

print("\nERROR HANDLING OK")
