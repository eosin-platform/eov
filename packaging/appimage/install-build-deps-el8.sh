#!/usr/bin/env bash
# Installs everything packaging/appimage/build.sh needs on an EL8 baseline
# (AlmaLinux/Rocky/RHEL 8, glibc 2.28). Release AppImages are built there so
# they run on every distribution with glibc 2.28 or newer; building on a
# current distribution would raise that floor to the build machine's glibc.
#
# Run as root inside the container. Rust itself is installed separately
# (rustup), so the same script serves CI and local `docker run` builds.
set -euo pipefail

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

. /etc/os-release
[[ "${PLATFORM_ID:-}" == "platform:el8" ]] || {
    echo "install-build-deps-el8.sh: expected an EL8 system, found ${PRETTY_NAME:-unknown}" >&2
    exit 1
}

dnf -y install dnf-plugins-core epel-release
dnf config-manager --set-enabled powertools
dnf -y install \
    bzip2 \
    cairo-devel \
    cpio \
    curl \
    diffutils \
    file \
    findutils \
    fontconfig-devel \
    freetype-devel \
    gcc \
    gcc-c++ \
    gdk-pixbuf2-devel \
    git-core \
    glib2-devel \
    gzip \
    libjpeg-turbo-devel \
    libpng-devel \
    libtiff-devel \
    libxml2-devel \
    libzstd-devel \
    make \
    ninja-build \
    openjpeg2-devel \
    patchelf \
    pkgconf-pkg-config \
    python3.11 \
    python3.11-pip \
    sqlite-devel \
    tar \
    which \
    xz \
    xz-devel \
    zlib-devel \
    zstd
# Libraries bundled into the AppImage carry their packages' licence files
# (build.sh collects them). Packages preinstalled in the container image were
# installed without documentation, where many keep those files; reinstall
# every package that ships a shared library to restore them.
mapfile -t library_packages < <(
    rpm -qf --queryformat '%{NAME}\n' /usr/lib64/*.so.* 2>/dev/null | grep -v 'not owned' | sort -u
)
dnf -y --setopt=tsflags= reinstall "${library_packages[@]}"
dnf clean all

# EL8's meson is too old for OpenSlide 4; install a pinned, hash-checked one.
python3.11 -m pip install --no-cache-dir --require-hashes --only-binary :all: \
    -r "$SCRIPT_DIR/requirements-build.txt"
meson --version
