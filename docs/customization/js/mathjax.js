// MathJax configuration for pymdownx.arithmatex (generic mode).
// arithmatex emits \( ... \) and \[ ... \] wrapped in .arithmatex spans/divs.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Re-typeset after Material for MkDocs instant navigation swaps page content.
if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise();
    }
  });
}
