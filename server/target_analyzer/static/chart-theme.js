/* Dark-theme defaults for Chart.js, matching the dashboard palette.
 *
 * Out of the box Chart.js assumes a light page: near-black tick/legend text and
 * a pale grid, both of which wash out on our dark surface. This sets the global
 * defaults once so every chart inherits them; per-dataset colours (the red/green
 * bars) stay where they are in each template.
 *
 * The values are read from the skin's variables on :root, so the charts follow
 * the stylesheet instead of restating it. Plain vendored JS, loaded after
 * chart.min.js and after the stylesheet <link>, no build step. Guarded so it's a
 * no-op if Chart failed to load. */
if (window.Chart) {
  const skin = getComputedStyle(document.documentElement);
  Chart.defaults.color = skin.getPropertyValue('--muted').trim(); // ticks + legend labels
  Chart.defaults.borderColor = skin.getPropertyValue('--line').trim(); // grid lines + axis borders
  Chart.defaults.font.family = skin.getPropertyValue('--sans').trim();
}
