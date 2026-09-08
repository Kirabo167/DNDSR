(presentations_page)=
# Presentations

Slide decks introducing DNDSR. Each deck is authored in Markdown and
rendered with [Marp](https://marp.app/); the CMake docs pipeline builds
the HTML and PDF outputs and stages them into this site under
`/presentations/`.

## DNDSR Overview (v0.2.0 historical snapshot)

A historical technical introduction for software and CS engineers, covering
architecture, the Geom pipeline, numerics, parallelism, I/O and
interop, solvers, engineering quality, results, and roadmap.  Available
in English and Chinese (中文).

The deck predates the v0.3.1 reactive-flow integration and the fork's ACM,
ACMVariable, and NCFV modules; its version-specific target/test statistics
must not be treated as the current support matrix. See the
[v0.3.1 feature and migration guide](../guides/v0.3.1_new_features_zh.md) and
the [current project structure](../guides/project_structure.md) for the active
codebase.

```{raw} html
<ul>
  <li><strong>Open slideshow (EN):</strong>
      <a href="../presentations/DNDSR_overview.html" target="_blank"
         rel="noopener">DNDSR_overview.html</a></li>
  <li><strong>Open slideshow (中文):</strong>
      <a href="../presentations/DNDSR_overview_zh.html" target="_blank"
         rel="noopener">DNDSR_overview_zh.html</a></li>
  <li><strong>Download PDF:</strong>
      <a href="../presentations/DNDSR_overview.pdf" target="_blank"
         rel="noopener">DNDSR_overview.pdf</a></li>
  <li><strong>Download PDF (中文):</strong>
      <a href="../presentations/DNDSR_overview_zh.pdf" target="_blank"
         rel="noopener">DNDSR_overview_zh.pdf</a></li>
  <li><strong>Source tree:</strong>
      <code>docs/presentations/DNDSR_overview/</code>
      (English parts in <code>parts/</code>, Chinese in <code>parts/zh/</code>)</li>
</ul>
```

Embedded preview (open in a new tab for fullscreen navigation):

```{raw} html
<iframe
    src="../presentations/DNDSR_overview.html"
    width="100%"
    height="600"
    style="border: 1px solid #d0d7de; border-radius: 6px;"
    title="DNDSR Overview Slide Deck"
    loading="lazy">
</iframe>
```

## Building the decks

Decks are rebuilt as part of the documentation target:

```sh
cmake --build build -t docs
```

To render a single deck manually (requires `marp-cli`):

```sh
bash docs/presentations/DNDSR_overview/build.sh --html --pdf             # English
bash docs/presentations/DNDSR_overview/build.sh --lang=zh --html --pdf   # Chinese
```

The build script assembles per-chapter Markdown fragments from `parts/`
(or `parts/zh/` for Chinese), copies image assets listed in
`res_manifest.txt` into `docs/presentations/res/`, pre-renders Mermaid
diagrams, and invokes Marp.  Slides marked with `<!-- _no_zh -->` are
automatically stripped from the Chinese edition.

See `docs/presentations/DNDSR_overview/README.md` for editing workflow,
overflow-detection tooling, and density-class conventions.
