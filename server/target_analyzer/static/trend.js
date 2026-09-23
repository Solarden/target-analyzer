/* The Trend line. Reads its data from the JSON island the template renders, so no
 * Jinja is interpolated into JavaScript.
 *
 * A category axis of pre-formatted date strings, not a time axis: Chart.js 4 time
 * scales need a date adapter, and nothing here is loaded from a CDN.
 *
 * Score and millimetres get their own y-axis — 0..100 and 0..200 on one scale makes
 * the smaller series a flat line at the bottom.
 *
 * Both axes are suggested from zero. Autoscaled to the span of the data, two sessions a
 * point apart fill the chart top to bottom and imply a trend that is not there. */
(function () {
  const island = document.getElementById("trend-data");
  const canvas = document.getElementById("trend");

  if (!island || !canvas || !window.Chart) return;

  const data = JSON.parse(island.textContent);

  new Chart(canvas, {
    type: "line",
    data: {
      labels: data.labels,
      datasets: [
        { label: "Total score", data: data.score, yAxisID: "score", borderColor: "#4f8cff", pointBackgroundColor: "#4f8cff" },
        { label: "Mean radius (mm)", data: data.mean_radius_mm, yAxisID: "mm", borderColor: "#3fb950", pointBackgroundColor: "#3fb950" },
        { label: "Extreme spread (mm)", data: data.extreme_spread_mm, yAxisID: "mm", borderColor: "#f0686b", pointBackgroundColor: "#f0686b", hidden: true },
        { label: "Bias (mm)", data: data.bias_mm, yAxisID: "mm", borderColor: "#9aa3b2", pointBackgroundColor: "#9aa3b2", hidden: true },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "index", intersect: false },
      // Chart.js fills a point with a near-transparent black, which on this background is
      // the difference between one session rendering and appearing not to.
      elements: { point: { radius: 4, hoverRadius: 6 } },
      scales: {
        score: {
          type: "linear",
          position: "left",
          suggestedMin: 0,
          title: { display: true, text: "score" },
        },
        mm: {
          type: "linear",
          position: "right",
          suggestedMin: 0,
          title: { display: true, text: "mm" },
          grid: { drawOnChartArea: false },
        },
      },
    },
  });
})();
