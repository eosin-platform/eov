cask "eov" do
    version "0.4.4"

    on_arm do
        sha256 "b1fb727b0d1f2daf3af3e3071c0011f0db9fd80f81ac553e61fb4c6db7e75c58"

        url "https://github.com/eosin-platform/eov/releases/download/v0.4.4/eov-v0.4.4-macos-arm64.zip"
    end
    on_intel do
        sha256 "7aa3a4646e4086549276a979b5f219857a661784a926f432fcff22abb9d6052d"

        url "https://github.com/eosin-platform/eov/releases/download/v0.4.4/eov-v0.4.4-macos-x86_64.zip"
    end

    name "eov"
    desc "Lightweight, cross-platform Whole Slide Image (WSI) viewer for digital pathology"
    homepage "https://eov.sh/"

    depends_on macos: :big_sur

    app "eov.app"
    binary "#{appdir}/eov.app/Contents/MacOS/eov"

    zap trash: [
        "~/Library/Application Support/io.eosin.eov",
        "~/Library/Caches/io.eosin.eov",
        "~/Library/Preferences/io.eosin.eov.plist",
    ]

    caveats <<~EOS
        eov is not notarized by Apple. If macOS prevents it from opening, go to:
            System Settings → Privacy & Security → Open Anyway
    EOS
end
