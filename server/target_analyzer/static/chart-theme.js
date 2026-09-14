/* Dark-theme defaults for Chart.js, matching the dashboard palette.
 *
 * Out of the box Chart.js assumes a light page: near-black tick/legend text and
 * a pale grid, both of which wash out on our dark surface (--color-bg #0f1115).
 * This sets the global defaults once so every chart inherits them; per-dataset
 * colours (the red/green bars) stay where they are in each template.
 *
 * Hex literals mirror the @theme tokens in tailwind.css (muted/line/font-sans) —
 * same pragmatic duplication the charts already make with #f0686b / #3fb950.
 * Plain vendored JS, loaded after chart.min.js, no build step (like the chart
 * library itself). Guarded so it's a no-op if Chart failed to load. */
if (window.Chart) {
  Chart.defaults.color = '#9aa3b2'; // --color-muted: ticks + legend labels
  Chart.defaults.borderColor = '#2a2f3a'; // --color-line: grid lines + axis borders
  Chart.defaults.font.family = 'system-ui, -apple-system, sans-serif'; // --font-sans
}
