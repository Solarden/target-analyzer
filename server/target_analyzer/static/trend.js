/* The Trend line. Reads its data from the JSON island the template renders, so no
 * Jinja is interpolated into JavaScript.
 *
 * A category axis of pre-formatted date strings, not a time axis: Chart.js 4 time
 * scales need a date adapter, and nothing here is loaded from a CDN.
 *
 * One line per shooter, for the one metric chosen above the chart. Each series has a
 * value only on its own shooter's rows, so spanGaps joins a shooter's sessions across
 * everyone else's.
 *
 * Suggested from zero. Autoscaled to the span of the data, two sessions a point apart
 * fill the chart top to bottom and imply a trend that is not there. */
(function () {
  const island = document.getElementById("trend-data");
  const canvas = document.getElementById("trend");

  if (!island || !canvas || !window.Chart) return;

  const data = JSON.parse(island.textContent);
  // The skin's categorical chart colours, read from its variables like chart-theme.js does.
  // A series keeps its slot under every filter, because its index is its shooter's place.
  const skin = getComputedStyle(document.documentElement);
  const colours = [1, 2, 3, 4, 5, 6, 7, 8].map((n) => skin.getPropertyValue(`--chart-${n}`).trim());
  // A shape per shooter as well as a colour, so colour alone never has to tell lines apart.
  const shapes = ["circle", "rect", "triangle", "rectRot", "star", "crossRot"];

  new Chart(canvas, {
    type: "line",
    data: {
      labels: data.labels,
      datasets: data.series.map((series) => {
        const colour = colours[series.colour % colours.length];
        const shape = shapes[series.colour % shapes.length];

        return {
          label: series.label,
          data: series.data,
          borderColor: colour,
          pointBackgroundColor: colour,
          pointStyle: shape,
        };
      }),
    },
    options: {
      responsive: true,
      spanGaps: true,
      interaction: { mode: "index", intersect: false },
      // Chart.js fills a point with a near-transparent black, which on this background is
      // the difference between one session rendering and appearing not to.
      elements: { point: { radius: 4, hoverRadius: 6 } },
      plugins: { legend: { labels: { usePointStyle: true } } },
      scales: {
        y: { type: "linear", suggestedMin: 0, title: { display: true, text: data.title } },
      },
    },
  });
})();
