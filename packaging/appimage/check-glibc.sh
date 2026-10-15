#!/usr/bin/env bash
# Reports the newest glibc symbol version any ELF file under a directory
# needs, and fails when it is newer than the supported baseline.
#
#   packaging/appimage/check-glibc.sh <dir-or-file> [max-version]
#
# The AppImage cannot bundle glibc, so the newest GLIBC_x.y symbol version
# needed by eov or any bundled library is the oldest glibc the AppImage runs
# on. Release builds use 2.28 (RHEL/Rocky/Alma 8, Debian 10, Ubuntu 18.10).
set -euo pipefail

root="${1:?usage: check-glibc.sh <dir-or-file> [max-version]}"
max_allowed="${2:-}"

command -v readelf >/dev/null 2>&1 || {
    echo "check-glibc: readelf is required" >&2
    exit 2
}

is_elf() {
    [[ -f "$1" && ! -L "$1" ]] || return 1
    [[ "$(head -c 4 "$1" | od -An -c | tr -d ' ')" == '177ELF' ]]
}

# Newest GLIBC_ version in the file's version-needs section.
newest_glibc() {
    readelf --version-info --wide "$1" 2>/dev/null \
        | awk '/Version needs section/ { needs = 1; next }
               /Version (definition|symbols) section/ { needs = 0 }
               needs { for (i = 1; i <= NF; i++) if ($i ~ /^GLIBC_[0-9]/) print $i }' \
        | sed 's/^GLIBC_//' \
        | sort -V \
        | tail -n 1
}

version_gt() {
    [[ "$1" != "$2" && "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n 1)" == "$1" ]]
}

overall=""
overall_file=""
failures=0
checked=0
while IFS= read -r -d '' file; do
    is_elf "$file" || continue
    checked=$((checked + 1))
    newest="$(newest_glibc "$file")"
    [[ -n "$newest" ]] || continue
    if [[ -z "$overall" ]] || version_gt "$newest" "$overall"; then
        overall="$newest"
        overall_file="$file"
    fi
    if [[ -n "$max_allowed" ]] && version_gt "$newest" "$max_allowed"; then
        echo "check-glibc: $file needs GLIBC_$newest (newer than GLIBC_$max_allowed)" >&2
        failures=$((failures + 1))
    fi
done < <(find "$root" -type f -print0)

if [[ "$checked" -eq 0 ]]; then
    echo "check-glibc: no ELF files found under $root" >&2
    exit 2
fi
if [[ -z "$overall" ]]; then
    echo "check-glibc: $checked ELF file(s), none needs a versioned glibc symbol"
else
    echo "check-glibc: $checked ELF file(s); newest requirement GLIBC_$overall ($overall_file)"
fi
if [[ "$failures" -gt 0 ]]; then
    echo "check-glibc: $failures file(s) need a glibc newer than $max_allowed" >&2
    exit 1
fi
