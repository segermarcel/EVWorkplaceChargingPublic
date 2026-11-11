const controls = Object.fromEntries(
  [...document.querySelectorAll("#parameters [id]")].map((element) => [element.id, element]),
);
const status = document.querySelector("#status");
const models = document.querySelector("#models");
const metricNames = {
  max_peak: ["Max. Peak", "kW"],
  charging_costs: ["Charging Costs", "€"],
  carbon_emissions: ["Carbon Emissions", "kg CO₂"],
};
let dataset;
let downloadUrls = [];

const format = (value) => new Intl.NumberFormat("en-GB", {
  maximumFractionDigits: 1, minimumFractionDigits: 1,
}).format(value);

function plotImage(company, period, model, portion, variant, description) {
  const path = `./plots/${company}_${period}_${model}_${portion}_11_${variant}.png`;
  return `<a class="plot-link" href="${path}" target="_blank" rel="noopener" aria-label="Open full-size ${description}"><img src="${path}" alt="${description}" width="1000" height="600" decoding="async"></a>`;
}

function valuesTable(result, model, absolute) {
  const mode = absolute ? "absolute" : "relative";
  const rows = Object.entries(metricNames).map(([key, [name, unit]]) =>
    `<tr><th scope="row">${name}</th><td data-metric="${key}" data-value="${result.metrics[key][mode]}">${format(result.metrics[key][mode])} ${absolute ? unit : "%"}</td></tr>`,
  ).join("");
  return `<table class="metric-table" aria-label="${dataset.models[model]} changes"><thead><tr><th scope="col">Change vs. uncontrolled charging</th><th scope="col">${absolute ? "Absolute" : "Relative"}</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function downloadCsv(result) {
  const fields = ["n", "Pb", "UCC", "Tc", "charging_costs", "grid_carbon_intensity"];
  const csv = [fields.join(","), ...result.profile.map((row) => fields.map((field) => row[field]).join(","))].join("\n") + "\n";
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  downloadUrls.push(url);
  return url;
}

function render() {
  if (!dataset) return;
  const weekly = controls.period.value === "week";
  controls.date.disabled = weekly;
  const company = controls.company.value, portion = controls.ev.value;
  const period = weekly ? "week" : controls.date.value;
  const selection = dataset.scenarios[`${company}_${portion}`];
  for (const url of downloadUrls) URL.revokeObjectURL(url);
  downloadUrls = [];
  const date = weekly ? "2024-02-01_2024-02-07" : period;
  models.innerHTML = Object.entries(dataset.models).map(([model, name]) => {
    const result = weekly ? selection[model].week : selection[model].days[date];
    const overlay = model === "ccm" && controls.electricity.checked
      || model === "cem" && controls.carbon.checked;
    const profileVariant = overlay ? "profile_overlay" : "profile";
    const metricsVariant = controls.absolute.checked ? "metrics_absolute" : "metrics_relative";
    const profileDescription = `${name}: ${weekly ? "mean daily" : "daily"} electricity profile${overlay ? model === "ccm" ? " with electricity costs" : " with grid carbon intensity" : ""}`;
    const metricsDescription = `${name}: ${controls.absolute.checked ? "absolute" : "relative"} changes compared with uncontrolled charging`;
    return `<section aria-labelledby="heading-${model}"><h2 id="heading-${model}">${name}</h2><div class="charts"><figure>${plotImage(company, period, model, portion, profileVariant, profileDescription)}</figure><figure>${plotImage(company, period, model, portion, metricsVariant, metricsDescription)}</figure></div><details class="values"><summary>Values &amp; download</summary>${valuesTable(result, model, controls.absolute.checked)}<a class="download" href="${downloadCsv(result)}" download="${company}_${date}_${model}_${portion}_11.csv">Download ${weekly ? "mean daily" : "daily"} profile (CSV)</a></details></section>`;
  }).join("");
  status.classList.remove("error");
  status.textContent = `Company ${company} · ${portion}% EVs · ${weekly ? "1–7 February 2024" : period}`;
  const note = document.querySelector("#period-note");
  note.hidden = false;
  note.textContent = weekly
    ? "Weekly view: mean daily profiles with 95% bootstrap confidence bands. Metrics are the mean of daily changes compared with uncontrolled charging; negative values indicate a reduction."
    : "Daily view: 15-minute electricity consumption. Metrics show the change compared with uncontrolled charging; negative values indicate a reduction.";
  for (const image of models.querySelectorAll("img")) {
    image.addEventListener("error", () => {
      if (!image.isConnected) return;
      status.textContent = "A chart could not be loaded. Please reload the page to try again.";
      status.classList.add("error");
    });
  }
  document.querySelector("main").setAttribute("aria-busy", "false");
  const query = new URLSearchParams({ company, ev: portion, period: controls.period.value, date: controls.date.value,
    electricity: String(Number(controls.electricity.checked)), carbon: String(Number(controls.carbon.checked)), absolute: String(Number(controls.absolute.checked)) });
  history.replaceState(null, "", `${location.pathname}?${query}${location.hash}`);
}

async function init() {
  try {
    const response = await fetch(new URL("./data/scenarios.json", import.meta.url));
    if (!response.ok) throw new Error(`Scenario download failed (${response.status})`);
    dataset = await response.json();
    for (const element of Object.values(controls)) element.disabled = false;
    for (const [id, options] of [["company", dataset.companies], ["ev", dataset.ev_portions]]) {
      controls[id].replaceChildren(...options.map((value) => new Option(value, value)));
      controls[id].disabled = false;
    }
    const query = new URLSearchParams(location.search);
    for (const [id, options] of [["company", dataset.companies], ["ev", dataset.ev_portions], ["period", ["day", "week"]], ["date", dataset.dates]]) {
      const value = query.get(id);
      if (options.map(String).includes(value)) controls[id].value = value;
    }
    for (const id of ["electricity", "carbon", "absolute"]) {
      controls[id].checked = query.has(id) ? query.get(id) === "1" : id !== "absolute" && controls.period.value === "day";
    }
    render();
    document.querySelector("#parameters").addEventListener("submit", (event) => event.preventDefault());
    document.querySelector("#parameters").addEventListener("change", (event) => {
      if (event.target === controls.date && !dataset.dates.includes(controls.date.value)) {
        controls.date.value = dataset.dates[0];
      }
      if (event.target === controls.period) {
        controls.electricity.checked = controls.carbon.checked = controls.period.value === "day";
      }
      render();
    });
  } catch (error) {
    status.textContent = "The scenarios could not be loaded. Please reload the page to try again.";
    status.classList.add("error");
    document.querySelector("main").setAttribute("aria-busy", "false");
    console.error(error);
  }
}

init();
