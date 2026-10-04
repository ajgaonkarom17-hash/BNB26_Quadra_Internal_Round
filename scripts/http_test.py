"""HTTP smoke test using urllib (no extra deps). Run the server first."""
import json
import os
import urllib.request

BASE = "http://127.0.0.1:" + os.environ.get("TL_PORT", "8000")


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path).read())


print("health:", get("/api/health"))
inv = post("/api/investigations", {"name": "URLLib Test"})
print("created:", inv["id"])

# multipart upload
boundary = "----trustlayer"
body = b""
body += f"--{boundary}\r\n".encode()
body += b'Content-Disposition: form-data; name="file"; filename="note.txt"\r\n'
body += b"Content-Type: text/plain\r\n\r\n"
body += b"Report confirmed 2020-01-01 at Riverside Park by Alex Morgan.\r\n"
body += f"--{boundary}--\r\n".encode()
req = urllib.request.Request(
    f"{BASE}/api/investigations/{inv['id']}/upload", data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
print("upload:", json.loads(urllib.request.urlopen(req).read()))

r = post(f"/api/investigations/{inv['id']}/analyze", {})
print("assessment:", r["uncertainty"]["final_assessment"], "coverage:", r["coverage"],
      "labels:", r["uncertainty"]["pre_fusion_assessment"])
print("cross pairs:", len(r["cross_modal"]["comparisons"]))
print("sample:", post("/api/sample", {})["investigation"]["name"])
print("list count:", len(get("/api/investigations")["investigations"]))
print("OK")
