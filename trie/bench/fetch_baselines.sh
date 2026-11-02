#!/usr/bin/env bash
# Fetch the optional baselines of the dictionary comparison (bench_dictcmp) at the commits the paper measured, and
# apply the patches this repository carries (trie/bench/adapters/*.patch). The build detects whatever is present
# (trie/cmake/Baselines.cmake); none of this is needed for the trie library itself.
#
#   usage: trie/bench/fetch_baselines.sh [--dir DIR] [NAME ...]
#     NAME   poplar | fst | fst64 | artsync | hot | c2   (default: all six)
#     --dir  where the checkouts go (default: trie/baselines, where CMake looks for poplar, fst, fst64 and artsync)
#
#   poplar   Poplar-trie (dynpdt-plain, dynpdt-compact)                    -> DIR/poplar-trie
#   fst      the upstream FST (fst-upstream)                                -> DIR/fast_succinct_trie
#   fst64    FST-64: a copy of the upstream FST's headers + FST64.patch     -> DIR/fast_succinct_trie_64
#   artsync  ARTSynchronized (art-olc) + the three artsync_olc_*.patch      -> DIR/ARTSynchronized
#   hot      HOT (hot-rowex)                                                -> DIR/hot      (-DTRIE_HOT_DIR=...)
#   c2       C2 (pdt-static) with lib/ds2i, its succinct, lib/sux, and
#            MARISA (marisa) at C2's baseline_marisa/marisa                 -> DIR/C2       (-DTRIE_C2_DIR=...)
#
# An existing directory is never modified: it is checked (commit, patches) and reported, so the script can be rerun.
# System dependencies, not fetched: the Boost headers (pdt-static; Boost.Hana for hot-rowex) and oneTBB (art-olc,
# hot-rowex; pass -DTBB_DIR=<prefix>/lib/cmake/TBB if CMake does not find it).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # trie/bench
PATCHES=$HERE/adapters
DIR=$(cd "$HERE/.." && pwd)/baselines
NAMES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dir) DIR=${2:?--dir needs a directory}; shift 2 ;;
    -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    poplar|fst|fst64|artsync|hot|c2) NAMES+=("$1"); shift ;;
    *) echo "fetch_baselines: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
done
[ ${#NAMES[@]} -gt 0 ] || NAMES=(poplar fst fst64 artsync hot c2)
mkdir -p "$DIR"
DIR=$(cd "$DIR" && pwd)
FAILED=0
CLONED=0

# The SHA-256 of FST64.patch is the fst-64 arm's identity (compiled into bench_dictcmp_fst64).
FST64_PATCH_SHA256=b44c29a258340ca24b57288a74a2f870ca2b34eba46e073557dfa2732c554bdd
ART_PATCHES=(artsync_olc_prefix_race.patch artsync_olc_scan_null_child.patch artsync_olc_scan_seek.patch)

say()  { printf '%-9s %s\n' "$1" "$2"; }
fail() { say "$1" "FAILED: $2"; FAILED=1; }
sha256() { if command -v sha256sum > /dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# clone <name> <url> <commit> <dir>: clone and check out the pinned commit; an existing checkout is only verified.
clone() {
  local name=$1 url=$2 commit=$3 dest=$4
  CLONED=0
  # an empty directory (e.g. an uninitialized submodule path of the parent checkout) counts as absent
  if [ -d "$dest" ] && [ -z "$(ls -A "$dest")" ]; then rmdir "$dest"; fi
  if [ -e "$dest" ]; then
    if [ "$(git -C "$dest" rev-parse --show-toplevel 2> /dev/null)" = "$(cd "$dest" && pwd -P)" ]; then
      local head; head=$(git -C "$dest" rev-parse HEAD)
      if [ "$head" = "$commit" ]; then say "$name" "present at the pinned commit ${commit:0:12} ($dest)"
      else fail "$name" "present at ${head:0:12}, pinned ${commit:0:12} ($dest left unchanged)"; return 1; fi
    else
      say "$name" "present, not a git checkout: not verified ($dest left unchanged)"
    fi
    return 0
  fi
  CLONED=1
  git clone --quiet "$url" "$dest" || { fail "$name" "git clone $url"; return 1; }
  git -C "$dest" -c advice.detachedHead=false checkout --quiet "$commit" || { fail "$name" "checkout $commit in $dest"; return 1; }
  say "$name" "cloned at ${commit:0:12} ($dest)"
}

# apply_patches <name> <dir> <patch>...: apply in order to a checkout this run just cloned.
apply_patches() {
  local name=$1 dest=$2; shift 2
  local p
  for p in "$@"; do
    git -C "$dest" apply "$PATCHES/$p" || { fail "$name" "$p does not apply in $dest"; return 1; }
    say "$name" "applied $p"
  done
}
# check_patches <name> <dir> <patch>...: an existing checkout is only checked: every patch must be applied, in order.
# Undo them from the last to the first in a scratch copy of the patched files (a patch that only adds lines would also
# "apply" a second time, so a forward --check proves nothing).
check_patches() {
  local name=$1 dest=$2; shift 2
  local tmp p i files=()
  for p in "$@"; do while read -r f; do files+=("$f"); done < <(sed -n 's#^+++ b/##p' "$PATCHES/$p"); done
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/fetch_baselines.XXXXXX")
  for f in "${files[@]}"; do mkdir -p "$tmp/$(dirname "$f")"; cp "$dest/$f" "$tmp/$f"; done
  for ((i = $# ; i >= 1 ; i--)); do
    p=${!i}
    if (cd "$tmp" && GIT_CEILING_DIRECTORIES="$(dirname "$tmp")" git apply --reverse "$PATCHES/$p" 2> /dev/null); then
      say "$name" "has $p"
    else
      rm -rf "$tmp"; fail "$name" "$p is not applied (in this order) in $dest (left unchanged)"; return 1
    fi
  done
  # undone, the files must be the pinned commit's, and no other tracked file may differ (catches a doubled patch)
  local f bad=0
  for f in $(printf '%s\n' "${files[@]}" | sort -u); do
    git -C "$dest" show "HEAD:$f" | cmp -s - "$tmp/$f" || bad=1
  done
  rm -rf "$tmp"
  local other; other=$(git -C "$dest" status --porcelain --untracked-files=no | sed 's/^...//' | grep -vxF -f <(printf '%s\n' "${files[@]}") || true)
  [ $bad = 0 ] || { fail "$name" "the patched files of $dest are not exactly the pinned commit plus the patches"; return 1; }
  [ -z "$other" ] || { fail "$name" "other local changes in $dest: $(echo $other)"; return 1; }
  say "$name" "patched files = pinned commit + the patches, exactly"
}

fetch_poplar() {
  clone poplar https://github.com/kampersanda/poplar-trie.git 4d8aaa9d1e84ba10dc338c1a5acc590bb6d86c50 "$DIR/poplar-trie"
}
fetch_fst() {
  clone fst https://github.com/kampersanda/fast_succinct_trie.git e0fcae0ae2fbcbaa448cc297b5bf28755bf159a4 "$DIR/fast_succinct_trie"
}
fetch_fst64() {
  local dest=$DIR/fast_succinct_trie_64
  if [ "$(sha256 "$PATCHES/FST64.patch")" != "$FST64_PATCH_SHA256" ]; then
    fail fst64 "$PATCHES/FST64.patch does not have the measured SHA-256 $FST64_PATCH_SHA256"; return 1
  fi
  if [ -e "$dest" ]; then
    if [ -f "$dest/FST64.patch" ] && [ "$(sha256 "$dest/FST64.patch")" = "$FST64_PATCH_SHA256" ]; then
      say fst64 "present with the measured FST64.patch ($dest)"
    else
      fail fst64 "present without the measured FST64.patch ($dest left unchanged)"; return 1
    fi
    return 0
  fi
  # the upstream checkout at the same commit is the source of the copy
  [ -d "$DIR/fast_succinct_trie/include" ] || fetch_fst || return 1
  local up=$DIR/fast_succinct_trie
  [ "$(git -C "$up" rev-parse HEAD)" = e0fcae0ae2fbcbaa448cc297b5bf28755bf159a4 ] || { fail fst64 "$up is not at the pinned commit"; return 1; }
  [ -z "$(git -C "$up" status --porcelain --untracked-files=no -- include LICENSE)" ] || { fail fst64 "$up has local changes in include/ or LICENSE"; return 1; }
  local tmp; tmp=$(mktemp -d "$DIR/.fst64.XXXXXX")
  cp -R "$up/include" "$up/LICENSE" "$tmp/"
  # applied as a plain patch: the ceiling keeps git from treating an enclosing repository (trie/ itself) as the root
  if ! (cd "$tmp" && GIT_CEILING_DIRECTORIES="$DIR" git apply "$PATCHES/FST64.patch"); then
    rm -rf "$tmp"; fail fst64 "FST64.patch does not apply"; return 1
  fi
  cp "$PATCHES/FST64.patch" "$tmp/FST64.patch"
  mv "$tmp" "$dest"
  say fst64 "created from the upstream FST e0fcae0ae2fb + FST64.patch ($dest)"
}
fetch_artsync() {
  local dest=$DIR/ARTSynchronized
  clone artsync https://github.com/flode/ARTSynchronized.git f00003ec7991da09c3dfc1c84da97c1965106eb3 "$dest" || return 1
  if [ "$CLONED" = 1 ]; then apply_patches artsync "$dest" "${ART_PATCHES[@]}"
  else check_patches artsync "$dest" "${ART_PATCHES[@]}"; fi
}
fetch_hot() {
  # HOT's own submodules (bundled TBB, Boost) are not needed: the adapter uses the system's oneTBB and Boost.Hana
  clone hot https://github.com/speedskater/hot.git 96bf6fb7103b27e50e16a6026db8974c090ee84a "$DIR/hot"
}
fetch_c2() {
  local c2=$DIR/C2
  clone c2 https://github.com/gt-alg-kit/C2.git 92c1b861456b2e5cb4085d317fc9ebb7c8e3d88e "$c2" || return 1
  # the PDT's dependencies and MARISA, cloned explicitly at their pins (C2's .gitmodules paths)
  clone c2/ds2i     https://github.com/aboffa/ds2i.git        4f46606cc07756bc5d919fc68fd7d567b901fb16 "$c2/lib/ds2i" || return 1
  local ok=0
  clone c2/succinct https://github.com/ot/succinct.git        ea74747a8b1cc7c934b984f23198efbb0635ed1f "$c2/lib/ds2i/succinct" || ok=1
  clone c2/sux      https://github.com/vigna/sux.git          568903f1b7957ef03620ebad65d6ef68e031fef9 "$c2/lib/sux" || ok=1
  clone c2/marisa   https://github.com/s-yata/marisa-trie.git 3e87d53b78e15f2f43783d5e376561a8c9722051 "$c2/baseline_marisa/marisa" || ok=1
  return $ok
}

for n in "${NAMES[@]}"; do "fetch_$n" || true; done

echo
echo "Configure with every fetched baseline (the configure output names what it found):"
args="cmake -S trie -B build -DCMAKE_BUILD_TYPE=Release"
[ "$DIR" = "$(cd "$HERE/.." && pwd)/baselines" ] || args+=" -DTRIE_POPLAR_DIR=$DIR/poplar-trie -DTRIE_FST_UPSTREAM_DIR=$DIR/fast_succinct_trie -DTRIE_FST64_DIR=$DIR/fast_succinct_trie_64 -DTRIE_ARTSYNC_DIR=$DIR/ARTSynchronized"
[ -d "$DIR/C2" ] && args+=" -DTRIE_C2_DIR=$DIR/C2"
[ -d "$DIR/hot" ] && args+=" -DTRIE_HOT_DIR=$DIR/hot"
echo "  $args   [-DTBB_DIR=<oneTBB prefix>/lib/cmake/TBB]"
exit $FAILED
