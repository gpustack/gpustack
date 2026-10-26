KaTeX 0.19.0
============

The public and bundled documentation share these unmodified KaTeX assets.
Keep fonts/ beside katex.min.css: the stylesheet uses relative font URLs.
LICENSE is the upstream MIT license, including the bundled fonts.

Source: https://registry.npmjs.org/katex/-/katex-0.19.0.tgz
npm metadata: https://registry.npmjs.org/katex/0.19.0
npm dist.integrity:
sha512-v6Tznz3zJ7u3niRCoDTsumM2+HA2XXcCu+WAacCeHD2z3p9A9Ks987o5FzfTGBN0e8A0vjEgIDvLbICrXpdw/Q==
The verified package files are recorded in tests/fixtures/katex-0.19.0.sha256.

To reproduce these files from the repository root (requires curl and shasum):

  (
  set -eu
  katex_tmp="$(mktemp -d)"
  trap 'rm -rf "$katex_tmp"' EXIT
  curl --fail --location \
    https://registry.npmjs.org/katex/-/katex-0.19.0.tgz \
    --output "$katex_tmp/katex.tgz"
  printf '%s  %s\n' \
    bfa4f39f3df327bbb79e2442a034ecba6336f870365d7702bbe58069c09e1c3db3de9f40f4ab3df3ba391737d31813747bc034be3120203bcb6c80ab5e9770fd \
    "$katex_tmp/katex.tgz" | shasum -a 512 --check
  tar -xzf "$katex_tmp/katex.tgz" -C "$katex_tmp" \
    package/dist/katex.min.css package/dist/katex.min.js \
    package/dist/contrib/auto-render.min.js package/dist/fonts package/LICENSE
  mkdir -p docs/assets/katex/contrib docs/assets/katex/fonts
  cp "$katex_tmp/package/dist/katex.min.css" docs/assets/katex/
  cp "$katex_tmp/package/dist/katex.min.js" docs/assets/katex/
  cp "$katex_tmp/package/dist/contrib/auto-render.min.js" docs/assets/katex/contrib/
  cp "$katex_tmp/package/dist/fonts/"* docs/assets/katex/fonts/
  cp "$katex_tmp/package/LICENSE" docs/assets/katex/
  )

When updating, select an explicit release and verify its npm dist.integrity
before replacing the version, source, checksum, scripts, stylesheet, and fonts.
Refresh the per-file SHA-256 manifest from that verified package as well.
