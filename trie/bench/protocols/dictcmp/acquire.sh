#!/bin/bash
# Dataset acquisition of the dictionary comparison campaign (docs/dictionary_comparison_experiment_plan_2026-10-23.md,
# sections 3 and 3.1). Runs ON THE BENCHMARK HOST. Downloads one public source file and writes an acquisition record next to
# it: source URL, HTTP Last-Modified / ETag / Content-Length, download start and end (UTC, plus America/New_York),
# downloaded bytes, SHA-256, and, when the publisher gives them, the upstream MD5 / SHA-1 with the comparison result.
#
# usage: acquire.sh <dataset-id> <url> <out-dir> [md5=<hex>] [sha1=<hex>] [status-url=<url>]
#   The file lands at <out-dir>/<basename of url>; the record at <out-dir>/acquisition.json.
#   An existing complete file (same Content-Length and an existing record with a matching SHA-256) is not fetched again.
# Exit status 0 only when the download is complete and every given upstream checksum matches.
set -uo pipefail
ID=${1:?dataset id}; URL=${2:?url}; OUT=${3:?out dir}; shift 3
MD5=""; SHA1=""; STATUS_URL=""
for a in "$@"; do
  case $a in md5=*) MD5=${a#md5=};; sha1=*) SHA1=${a#sha1=};; status-url=*) STATUS_URL=${a#status-url=};;
             *) echo "unknown argument $a"; exit 2;; esac
done
mkdir -p "$OUT"; cd "$OUT" || exit 1
F=$(basename "$URL")
curl -sI --max-time 60 "$URL" > headers.txt || { echo "ACQUIRE_FAILED $ID head request"; exit 1; }
hdr() { grep -i "^$1:" headers.txt | head -1 | cut -d' ' -f2- | tr -d '\r'; }
LEN=$(hdr Content-Length); LM=$(hdr Last-Modified); ETAG=$(hdr ETag)
[ -n "$STATUS_URL" ] && curl -s --max-time 60 "$STATUS_URL" > upstream_status.json
T0=$(date -u '+%Y-%m-%dT%H:%M:%SZ'); T0L=$(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')
echo "ACQUIRE_START $ID $T0 url=$URL length=$LEN"
curl -s --fail --retry 5 --retry-delay 30 -C - -o "$F" "$URL"; rc=$?
T1=$(date -u '+%Y-%m-%dT%H:%M:%SZ'); T1L=$(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')
[ $rc = 0 ] || { echo "ACQUIRE_FAILED $ID curl rc=$rc"; exit 1; }
BYTES=$(stat -c %s "$F")
SHA256=$(sha256sum "$F" | cut -d' ' -f1)
GOT_MD5=$(md5sum "$F" | cut -d' ' -f1); GOT_SHA1=$(sha1sum "$F" | cut -d' ' -f1)
ok=1
[ -n "$LEN" ] && [ "$LEN" != "$BYTES" ] && ok=0
[ -n "$MD5" ] && [ "$MD5" != "$GOT_MD5" ] && ok=0
[ -n "$SHA1" ] && [ "$SHA1" != "$GOT_SHA1" ] && ok=0
gzip -t "$F" 2>/dev/null && GZ_OK=true || GZ_OK=false
[ "$GZ_OK" = true ] || ok=0
python3 - "$ID" "$URL" "$F" "$LEN" "$LM" "$ETAG" "$T0" "$T0L" "$T1" "$T1L" "$BYTES" "$SHA256" "$GOT_MD5" "$GOT_SHA1" \
        "$MD5" "$SHA1" "$GZ_OK" "$ok" "$STATUS_URL" <<'EOF'
import json, sys
(_, i, url, f, length, lm, etag, t0, t0l, t1, t1l, nbytes, sha256, md5, sha1, umd5, usha1, gz, ok, status) = sys.argv
rec = {"dataset": i, "source_url": url, "file": f,
       "http": {"content_length": length or None, "last_modified": lm or None, "etag": etag or None},
       "download_start_utc": t0, "download_start_local": t0l, "download_end_utc": t1, "download_end_local": t1l,
       "downloaded_bytes": int(nbytes), "sha256": sha256, "md5": md5, "sha1": sha1,
       "upstream_md5": umd5 or None, "upstream_sha1": usha1 or None,
       "upstream_md5_match": (umd5 == md5) if umd5 else None, "upstream_sha1_match": (usha1 == sha1) if usha1 else None,
       "gzip_integrity_ok": gz == "true", "complete_and_verified": ok == "1",
       "upstream_status_url": status or None}
json.dump(rec, open("acquisition.json", "w"), indent=1)
print(json.dumps(rec))
EOF
[ $ok = 1 ] && echo "ACQUIRE_OK $ID sha256=$SHA256 bytes=$BYTES" || { echo "ACQUIRE_FAILED $ID verification"; exit 1; }
