"use strict";

// Every displayed observation comes from the local canonical experiment API.
const state = {runs: [], metrics: [], evaluations: [], resources: [], controls: [], efficiency: [], efficiencyComparisons: [], efficiencyError: "", efficiencyStudy: "", efficiencyLoaded: false, view: "overview", metric: "", axis: "elapsed_seconds", search: "", sort: "start_time", simulator: "", minimumDuration: 0, after: "", threshold: "", resourcesTruncated: false, controlsTruncated: false};
const palette = ["#248a6b", "#7f94b8", "#cb9c57", "#a47ca4", "#6ba6aa", "#aab568", "#cc7e73", "#657d60", "#7585b9", "#b39582", "#73ac91", "#985e78"];
const titles = {
  overview: ["Learning, measured.", "Track quality, experience and time from reproducible experiment records.", "Overview"],
  runs: ["Every result has a history.", "Inspect source, configuration, compute, checkpoints and model ancestry.", "Runs & lineage"],
  learning: ["The learning-efficiency frontier.", "Compare quality with elapsed time, experience and measured resource use.", "Learning & efficiency"],
  "real-efficiency": ["Real competence efficiency.", "Real capability determines value. Experience, compute cost and elapsed time remain separate.", "Real competence efficiency"],
  benchmark: ["One hour. A governed test.", "Checkpoint evidence remains separate from longer training and real-game qualification.", "One-hour benchmark"],
  transfer: ["Beyond the training condition.", "Inspect held-out vehicle and map behavior, adaptation and the sim-to-real gap.", "Transfer & adaptation"],
  controls: ["Observe the dynamics.", "Inspect independent pedal channels, action order and calibration evidence.", "Controls & calibration"]
};
const $ = selector => document.querySelector(selector);
const esc = value => String(value ?? "").replace(/[&<>"']/g, character => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[character]));
const finite = value => typeof value === "number" && Number.isFinite(value);
const num = (value, digits = 1) => finite(value) ? value.toLocaleString(undefined, {maximumFractionDigits: digits}) : "Not measured";
const compact = value => finite(value) ? Intl.NumberFormat(undefined, {notation: "compact", maximumFractionDigits: 1}).format(value) : "Not measured";
const seconds = value => !finite(value) ? "Not measured" : value >= 3600 ? `${num(value / 3600, 2)} h` : value >= 60 ? `${num(value / 60, 1)} min` : `${num(value, 1)} s`;
const date = value => value ? new Date(value).toLocaleString(undefined, {month:"short", day:"numeric", hour:"2-digit", minute:"2-digit"}) : "Not measured";
const pretty = value => esc(JSON.stringify(value, null, 2));
const empty = (message = "Not measured", explanation = "This view will populate when a governed run records this evidence.", compactView = false) => `<div class="empty ${compactView ? "compact" : ""}"><strong>${esc(message)}</strong><p>${esc(explanation)}</p></div>`;
const badge = status => `<span class="badge ${esc(status)}">${esc(status)}</span>`;
const panel = (title, subtitle, body, tag = "") => `<section class="panel"><div class="panel-header"><div><h2>${esc(title)}</h2>${subtitle ? `<p>${esc(subtitle)}</p>` : ""}</div>${tag ? `<span class="tag">${esc(tag)}</span>` : ""}</div>${body}</section>`;
const stat = (label, value, note) => `<div class="stat"><div class="stat-label">${esc(label)}</div><div class="stat-value ${String(value).length > 14 ? "textual" : ""}">${esc(value)}</div><div class="stat-note">${esc(note)}</div></div>`;
const table = (headers, rows) => rows.length ? `<div class="table-wrap"><table><thead><tr>${headers.map(header => `<th>${esc(header)}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join("")}</tr>`).join("")}</tbody></table></div>` : empty("No matching records", "Adjust the filters or run an experiment to populate this view.", true);
const runName = run => `${run.algorithm} · seed ${run.seed}`;
const runButton = run => `<button class="row-link" data-run="${esc(run.run_id)}">${esc(runName(run))}</button><span class="small-id">${esc(run.run_id.slice(0, 8))} · ${esc(run.experiment_id)}</span>`;
const byId = id => state.runs.find(run => run.run_id === id);
const metricLabel = name => name.replaceAll("_", " ").replaceAll("/", " / ");
const domain = run => run?.metadata?.evidence_domain || (run?.environment === "actual_hill_climb_racing" ? "real_game" : run?.environment === "uncalibrated_hill_surrogate" ? "uncalibrated simulation" : run?.environment?.includes("synthetic") ? "synthetic demonstration" : run?.environment || "unspecified environment");

function scopedRuns() {
  const selected = {algorithm: $("#algorithm").value, environment: $("#environment").value, status: $("#status").value, vehicle_profile: $("#vehicle").value, map_profile: $("#map").value};
  return state.runs.filter(run => Object.entries(selected).every(([key, value]) => !value || run[key] === value)).filter(run => !state.search || `${run.run_id} ${run.experiment_id} ${run.algorithm} ${JSON.stringify(run.policy_architecture)} ${JSON.stringify(run.summary?.model_config)}`.toLowerCase().includes(state.search.toLowerCase())).filter(run => !state.simulator || run.simulator_version === state.simulator).filter(run => !state.after || run.start_time >= state.after).filter(run => !state.minimumDuration || (run.duration ?? 0) >= state.minimumDuration);
}
function scope() {
  const runs = scopedRuns(), ids = new Set(runs.map(run => run.run_id));
  return {runs, metrics: state.metrics.filter(row => ids.has(row.run_id)), evaluations: state.evaluations.filter(row => ids.has(row.run_id)), resources: state.resources.filter(row => ids.has(row.run_id)), controls: state.controls.filter(row => ids.has(row.run_id))};
}
function quantile(values, q) {
  if (!values.length) return null;
  const ordered = [...values].sort((a,b) => a-b), position = (ordered.length - 1) * q, low = Math.floor(position);
  return ordered[low] + (ordered[Math.ceil(position)] - ordered[low]) * (position - low);
}
function selectedMetric(metrics) {
  const available = [...new Set(metrics.map(row => row.name))].sort();
  if (!available.includes(state.metric)) state.metric = ["mean_episode_distance", "median_distance", "quality", "loss", "environment_steps_per_second"].find(name => available.includes(name)) || available[0] || "";
  return available;
}
function metricControls(metrics, includeThreshold = false) {
  const available = selectedMetric(metrics);
  return `<div class="local-controls"><label>Measurement<select id="metric">${available.map(name => `<option value="${esc(name)}" ${name === state.metric ? "selected" : ""}>${esc(metricLabel(name))}</option>`).join("")}</select></label><label>Horizontal axis<select id="axis"><option value="elapsed_seconds" ${state.axis === "elapsed_seconds" ? "selected" : ""}>Wall-clock seconds</option><option value="step" ${state.axis === "step" ? "selected" : ""}>Environment / optimizer steps</option><option value="episodes" ${state.axis === "episodes" ? "selected" : ""}>Episodes</option></select></label>${includeThreshold ? `<label>Time to threshold (≥)<input id="threshold" type="number" step="any" value="${esc(state.threshold)}" placeholder="Enter a target"></label>` : ""}</div>`;
}
function xValue(row) { return state.axis === "episodes" ? row.dimensions?.episodes : row[state.axis]; }
function metricSeries(metrics, name = state.metric) {
  const groups = new Map();
  for (const row of metrics.filter(row => row.name === name)) {
    if (!finite(xValue(row)) || !finite(row.value)) continue;
    if (!groups.has(row.run_id)) groups.set(row.run_id, []);
    groups.get(row.run_id).push({x: xValue(row), y: row.value});
  }
  return [...groups].map(([id, points], index) => ({id, name: byId(id) ? `${runName(byId(id))} · ${id.slice(0,6)}` : id.slice(0,8), points: points.sort((a,b) => a.x-b.x), color: palette[index % palette.length]}));
}
function medianStepSeries(data) {
  const groups=new Map();
  for(const row of data.metrics.filter(row=>row.name===state.metric&&byId(row.run_id)?.status==="completed")){
    const run=byId(row.run_id),key=`${run.algorithm} · ${run.environment}`;
    if(!groups.has(key))groups.set(key,new Map());
    const steps=groups.get(key);if(!steps.has(row.step))steps.set(row.step,new Map());
    steps.get(row.step).set(row.run_id,row.value);
  }
  return [...groups].map(([name,steps],index)=>({name,color:palette[index%palette.length],points:[...steps].filter(([,values])=>values.size>=2).map(([step,values])=>({x:step,y:quantile([...values.values()],.5),low:quantile([...values.values()],.25),high:quantile([...values.values()],.75),n:values.size})).sort((a,b)=>a.x-b.x)}));
}
function chart(series, {xLabel = "Elapsed seconds", yLabel = "", zero = false, scatter = false, stepped = false} = {}) {
  const populated = series.filter(item => item.points.some(point => finite(point.x) && finite(point.y)));
  if (!populated.length) return empty("Not measured", "No observations with both coordinates are available for the selected scope.");
  // A bounded series count preserves readability; actual exported rows remain complete.
  const drawn = populated.slice(0, 12), points = drawn.flatMap(item => item.points).filter(point => finite(point.x) && finite(point.y));
  const width = 720, height = 270, left = 54, right = 14, top = 13, bottom = 38;
  let xMin = Math.min(...points.map(point => point.x)), xMax = Math.max(...points.map(point => point.x));
  let yMin = zero ? Math.min(0, ...points.map(point => point.low ?? point.y)) : Math.min(...points.map(point => point.low ?? point.y)), yMax = Math.max(...points.map(point => point.high ?? point.y));
  if (xMin === xMax) { xMin = Math.min(0, xMin); xMax = Math.max(xMax + 1, 1); }
  const pad = (yMax-yMin || Math.abs(yMax)*0.1 || 1)*0.1;
  yMin -= zero && yMin === 0 ? 0 : pad; yMax += pad;
  const x = value => left + (value-xMin)/(xMax-xMin)*(width-left-right), y = value => height-bottom-(value-yMin)/(yMax-yMin)*(height-top-bottom);
  const grid = Array.from({length:5}, (_,i) => { const value = yMin + (yMax-yMin)*i/4; return `<line class="gridline" x1="${left}" x2="${width-right}" y1="${y(value)}" y2="${y(value)}"/><text x="${left-9}" y="${y(value)+3}" text-anchor="end">${esc(compact(value))}</text>`; }).join("");
  const ticks = Array.from({length:5}, (_,i) => { const value = xMin + (xMax-xMin)*i/4; return `<text x="${x(value)}" y="${height-bottom+19}" text-anchor="middle">${esc(compact(value))}</text>`; }).join("");
  const marks = drawn.map(item => {
    const valid = item.points.filter(point => finite(point.x) && finite(point.y));
    // Decimate only for rendering, retaining first/last; source exports are unchanged.
    const stride = Math.max(1, Math.ceil(valid.length/1800)), sampled = valid.filter((_,i) => i % stride === 0 || i === valid.length-1);
    const line = !scatter && sampled.length > 1 ? `<path class="line" stroke="${item.color}" d="${sampled.map((point,i) => `${i ? stepped ? `H${x(point.x).toFixed(2)} V` : "L" : "M"}${i && stepped ? y(point.y).toFixed(2) : `${x(point.x).toFixed(2)},${y(point.y).toFixed(2)}`}`).join(" ")}"/>` : "";
    const band=sampled.length>1&&sampled.every(point=>finite(point.low)&&finite(point.high))?`<path fill="${item.color}22" stroke="none" d="${sampled.map((point,i)=>`${i?"L":"M"}${x(point.x)},${y(point.high)}`).join(" ")} ${[...sampled].reverse().map(point=>`L${x(point.x)},${y(point.low)}`).join(" ")} Z"/>`:"";
    const dots = (scatter ? sampled : sampled.length < 35 ? sampled : [sampled[0],sampled.at(-1)]).map(point => `<circle cx="${x(point.x)}" cy="${y(point.y)}" r="${scatter ? 4 : 2.5}" fill="${item.color}"><title>${esc(item.name)}: ${esc(num(point.y,4))} at ${esc(num(point.x,2))}${point.n?` (${point.n} runs)`:""}</title></circle>`).join("");
    return band + line + dots;
  }).join("");
  return `<div class="chart"><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(yLabel || state.metric)} plotted against ${esc(xLabel)}">${grid}${ticks}${marks}<text x="${width-right}" y="${height-2}" text-anchor="end">${esc(xLabel)}</text></svg></div><div class="legend">${drawn.map(item => `<span><i style="background:${item.color}"></i>${esc(item.name)}</span>`).join("")}</div>${populated.length > 12 ? `<p class="context">Showing 12 of ${populated.length} series. Narrow the run filters to inspect the remaining series.</p>` : ""}`;
}
const axisLabel = () => ({elapsed_seconds:"Wall-clock seconds", step:"Recorded steps", episodes:"Episodes"}[state.axis]);
function runTable(runs, limit = null) {
  const sorted = [...runs].sort((a,b) => state.sort === "duration" ? (b.duration ?? -1)-(a.duration ?? -1) : state.sort === "algorithm" ? a.algorithm.localeCompare(b.algorithm) : b.start_time.localeCompare(a.start_time));
  const shown = limit ? sorted.slice(0,limit) : sorted;
  return table(["Run / seed", "Environment", "Status", "Duration", "Steps", "Started"], shown.map(run => [runButton(run), `${esc(run.environment)}<span class="model-qualifier">${esc([run.vehicle_profile,run.map_profile].filter(Boolean).join(" / ") || domain(run))}</span>`, badge(run.status), seconds(run.duration), run.status === "running" ? "Not finalized" : compact(run.environment_steps), esc(date(run.start_time))]));
}
function finalMetrics(metrics) {
  const latest = new Map();
  for (const row of metrics.filter(row => row.name === state.metric && byId(row.run_id)?.status === "completed")) if (!latest.has(row.run_id) || latest.get(row.run_id).elapsed_seconds <= row.elapsed_seconds) latest.set(row.run_id, row);
  return [...latest.values()];
}
function quality(evaluation) {
  const result = evaluation.results || {};
  const candidates = [["Mean distance", result.mean_distance], ["Mean distance", result.summary?.distance?.mean], ["Quality", result.quality], ["Absolute error", result.absolute_error]];
  return candidates.find(([,value]) => finite(value)) || ["Quality", null];
}
function trainingRun(row) {
  const evaluationRun=byId(row.run_id), parent=byId(evaluationRun?.parent_run);
  const owns=run=>row.checkpoint_hash && (run?.artifact_manifest||[]).some(artifact=>artifact.kind==="checkpoint"&&artifact.sha256===row.checkpoint_hash);
  if (owns(parent)) return parent;
  const owners=state.runs.filter(owns);
  return owners.length===1 ? owners[0] : null;
}
const trainingClass = row => trainingRun(row)?.metadata?.benchmark_class || byId(row.run_id)?.metadata?.benchmark_class || "Not recorded";
function checkpointTargetMinutes(row) {
  if (finite(row.metadata?.requested_minutes)) return row.metadata.requested_minutes;
  const artifact=trainingRun(row)?.artifact_manifest?.find(item=>item.kind==="checkpoint"&&item.sha256===row.checkpoint_hash);
  return finite(artifact?.metadata?.target_seconds) ? artifact.metadata.target_seconds/60 : null;
}
const checkpointActualSeconds = row => finite(row.metadata?.training_elapsed_seconds) ? row.metadata.training_elapsed_seconds : finite(row.metadata?.training_minutes) ? row.metadata.training_minutes*60 : null;
function coldStartHour(row) {
  const owner=trainingRun(row);
  return row.protocol?.startsWith("one-hour") && byId(row.run_id)?.status==="completed" && owner?.status==="completed" && trainingClass(row)==="cold_start" && owner.configuration?.seconds===3600 && !owner.parent_checkpoint && !owner.configuration?.parent_checkpoint;
}
const runCondition = run => ({profile:run?.configuration?.profile || run?.vehicle_profile,terrain:run?.configuration?.terrain || run?.map_profile});
function trainedCondition(row) {
  const config=row.results?.policy_config || {}, owner=runCondition(trainingRun(row));
  return {profile:config.profile || owner.profile,terrain:config.terrain || owner.terrain};
}
const evaluatedCondition = row => ({profile:row.results?.profile || row.metadata?.vehicle_profile,terrain:row.results?.terrain || row.metadata?.map_profile});
const conditionName = condition => `${condition.profile || "Not recorded"} / ${condition.terrain || "Not recorded"}`;
const sameCondition = (a,b) => a.profile && a.terrain && b.profile && b.terrain && a.profile===b.profile && a.terrain===b.terrain;
function adaptedTraining(row) {
  const owner=trainingRun(row), parent=byId(owner?.parent_run), current=runCondition(owner), previous=runCondition(parent);
  return owner?.status==="completed" && parent && current.profile && current.terrain && previous.profile && previous.terrain && !sameCondition(current,previous) && ["fine_tuning","generalist_adaptation"].includes(trainingClass(row));
}
function adaptationMinutes(row) {
  if (byId(row.run_id)?.status!=="completed") return null;
  if (finite(row.metadata?.adaptation_minutes)) return row.metadata.adaptation_minutes;
  return adaptedTraining(row) && finite(checkpointActualSeconds(row)) ? checkpointActualSeconds(row)/60 : null;
}
function evaluationContext(row) {
  if (!adaptedTraining(row)) return "";
  if (sameCondition(evaluatedCondition(row),trainedCondition(row))) return "Adapted condition";
  if (sameCondition(evaluatedCondition(row),runCondition(byId(trainingRun(row)?.parent_run)))) return "Original-condition retention";
  return "Post-adaptation evaluation";
}
function overview(data) {
  selectedMetric(data.metrics);
  const completed = data.runs.filter(run => run.status === "completed").length;
  const training = data.runs.reduce((total,run) => total + (run.summary?.training_clock_seconds ?? run.wall_clock_seconds ?? 0),0);
  const steps = data.runs.reduce((total,run) => total + (run.environment_steps ?? 0),0);
  const real = data.evaluations.filter(row => ["real_game", "real-game"].includes(domain(byId(row.run_id))));
  const stats = `<div class="stats">${stat("Recorded experiments", String(data.runs.length), `${completed} completed · ${data.runs.filter(run=>run.status==='running').length} running`)}${stat("Recorded environment steps", data.runs.length ? compact(steps) : "Not measured", "Experience in the selected run population")}${stat("Total recorded time", data.runs.length ? seconds(training) : "Not measured", "Training clocks where recorded; otherwise run duration")}${stat("Real-game evaluation", real.length ? `${real.length} evaluations` : "Not measured", "Simulator scores do not establish real-game competence")}</div>`;
  const curves = metricControls(data.metrics) + chart(metricSeries(data.metrics), {xLabel:axisLabel()});
  const evaluations = data.evaluations.slice().sort((a,b)=>b.timestamp.localeCompare(a.timestamp)).slice(0,5);
  const evidence = evaluations.length ? `<ul class="metric-list">${evaluations.map(row => { const run=byId(row.run_id), [label,value]=quality(row); return `<li><div>${run ? runButton(run) : esc(row.run_id.slice(0,8))}<span class="model-qualifier">${esc(label)} · ${esc(row.results?.scope || domain(run))}</span></div><strong>${num(value,2)}</strong></li>`; }).join("")}</ul>` : empty("No evaluation records", "Training curves and held-out evaluations answer different questions.");
  return stats + `<div class="grid two">${panel("Quality over experience", "Observed points for each run; gaps are not imputed.", curves, "Run comparison")}${panel("Latest evaluation records", "Separate evidence by environment and declared evaluation protocol.", evidence, `${data.evaluations.length} records`)}</div>` + panel("Recent experiments", "Select a run to inspect its complete provenance and artifact manifest.", runTable(data.runs,8), `${data.runs.length} in scope`);
}
function runsView(data) {
  const versions = [...new Set(state.runs.map(run=>run.simulator_version).filter(Boolean))].sort();
  const controls = `<div class="local-controls"><label>Search model / run<input id="search" type="search" value="${esc(state.search)}" placeholder="Algorithm, model, ID…"></label><label>Simulator<select id="simulator"><option value="">All versions</option>${versions.map(version=>`<option ${state.simulator===version?"selected":""}>${esc(version)}</option>`).join("")}</select></label><label>Started on or after<input id="after" type="date" value="${esc(state.after)}"></label><label>Minimum duration (seconds)<input id="minimum-duration" type="number" min="0" value="${state.minimumDuration || ""}" placeholder="Any duration"></label><label>Sort by<select id="sort"><option value="start_time" ${state.sort==="start_time"?"selected":""}>Most recent</option><option value="duration" ${state.sort==="duration"?"selected":""}>Longest duration</option><option value="algorithm" ${state.sort==="algorithm"?"selected":""}>Algorithm</option></select></label></div>`;
  const children = data.runs.filter(run=>run.parent_run || run.parent_checkpoint);
  const lineage = children.length ? `<div class="lineage">${children.map(run=>`<div class="lineage-node"><strong>${runButton(run)}</strong>${esc(run.metadata?.benchmark_class || "Derived model")}<div class="parent">Parent run: ${esc(run.parent_run || "Not recorded")}<br>Parent checkpoint: ${esc(run.parent_checkpoint || "Not recorded")}</div></div>`).join("")}</div>` : empty("No parent–child lineage yet", "Independent initializations have no declared parent checkpoint.");
  return `<div class="grid">${panel("Run catalog", "Filters also scope the charts and exported evidence in all views.", controls+runTable(data.runs))}${panel("Model lineage", "Explicit parent references establish ancestry; matching algorithm names do not.",lineage)}</div>`;
}
function learning(data) {
  selectedMetric(data.metrics);
  const grouped = new Map();
  for (const row of finalMetrics(data.metrics)) {const run=byId(row.run_id), key=`${run.algorithm} · ${run.environment}`; if(!grouped.has(key)) grouped.set(key,[]); grouped.get(key).push({row,run});}
  const comparisons = table(["Algorithm / environment", "Runs / seeds", "Median final", "25th–75th percentile", "Range", "Median time to target"],[...grouped].map(([key, entries])=>{const values=entries.map(({row})=>row.value), seeds=new Set(entries.map(({run})=>run.seed)), attained=state.threshold!=="" ? entries.map(({run})=>data.metrics.filter(row=>row.run_id===run.run_id && row.name===state.metric && row.value>=Number(state.threshold)).sort((a,b)=>a.elapsed_seconds-b.elapsed_seconds)[0]?.elapsed_seconds).filter(finite) : []; return [esc(key),`${entries.length} / ${seeds.size}`,num(quantile(values,.5),3),`${num(quantile(values,.25),3)} – ${num(quantile(values,.75),3)}`,`${num(Math.min(...values),3)} – ${num(Math.max(...values),3)}`,state.threshold===""?"No target set":attained.length?`${seconds(quantile(attained,.5))} (${attained.length}/${entries.length} attained)`:"Not attained"]; }));
  const resourceSeries = field => data.runs.map((run,index)=>({name:`${runName(run)} · ${run.run_id.slice(0,6)}`,color:palette[index%palette.length],points:data.resources.filter(row=>row.run_id===run.run_id&&finite(row[field])).map(row=>({x:row.elapsed_seconds,y:row[field]}))}));
  const frontier = [...grouped].map(([name,entries],index)=>({name,color:palette[index%palette.length],points:entries.map(({run,row})=>({x:run.summary?.training_clock_seconds ?? run.duration,y:row.value}))}));
  const latencyNames=["inference_seconds","optimizer_seconds","environment_seconds","environment_steps_per_second"];
  const latencies=data.runs.filter(run=>data.metrics.some(row=>row.run_id===run.run_id&&latencyNames.includes(row.name))).map(run=>[runButton(run),...latencyNames.map(name=>{const latest=data.metrics.filter(row=>row.run_id===run.run_id&&row.name===name).sort((a,b)=>b.elapsed_seconds-a.elapsed_seconds)[0];return name==="environment_steps_per_second"?num(latest?.value,1):seconds(latest?.value);})]);
  return panel("Learning curves", "Metric points use their recorded coordinates. No cross-seed interpolation is applied.", metricControls(data.metrics,true)+chart(metricSeries(data.metrics),{xLabel:axisLabel()})) + `<div class="grid" style="margin-top:20px">${panel("Algorithm and seed comparison", "Completed runs only, grouped by algorithm and environment. Settings may vary. Percentiles describe final observations; repeated seeds are not independent replicates.",comparisons)}${panel("Median curve at shared recorded steps", "At least two completed runs must share a step. Shading is the middle 50% of observed run values, not a confidence interval. Settings may vary.",chart(medianStepSeries(data),{xLabel:"Recorded steps"}))}${panel("Quality versus training time", "Each dot is one completed run; the envelope supports inspection of the efficiency frontier, not causal claims.",chart(frontier,{xLabel:"Training / run seconds",scatter:true}))}</div><div class="grid equal">${panel("CPU utilization", "Sampled host utilization, percent.",chart(resourceSeries("cpu_percent"),{zero:true,yLabel:"CPU percent"}),state.resourcesTruncated?"First 10,000 samples":"Recorded telemetry")}${panel("GPU utilization", "Unavailable samples remain missing.",chart(resourceSeries("gpu_percent"),{zero:true,yLabel:"GPU percent"}))}</div>` + panel("Measured runtime costs", "Latest cumulative component times per run. Per-call inference latency has not been recorded.",table(["Run","Inference total","Optimizer total","Environment total","Latest steps / second"],latencies));
}
function benchmark(data) {
  const targets=[5,10,20,30,45,60];
  const allRecords=data.evaluations.filter(row=>row.protocol?.startsWith("one-hour")&&byId(row.run_id)?.status==="completed");
  const records=allRecords.filter(coldStartHour), continued=allRecords.filter(row=>!coldStartHour(row));
  const measured=new Set(records.map(checkpointTargetMinutes));
  const rail=`<div class="progress-rail">${targets.map(minute=>`<div class="checkpoint ${measured.has(minute)?"measured":""}"><b>${minute}<small>minutes</small></b><small>${measured.has(minute)?"Evaluation recorded":"Not measured"}</small></div>`).join("")}</div>`;
  const evaluationRows=entries=>entries.map(row=>{const run=byId(row.run_id),[label,value]=quality(row);return [run ? runButton(run):esc(row.run_id.slice(0,8)),num(checkpointTargetMinutes(row),2),finite(checkpointActualSeconds(row))?`${num(checkpointActualSeconds(row),2)} s`:"Not measured",esc(domain(run)),`${num(value,2)}<span class="model-qualifier">${esc(label)}</span>`,esc(row.episodes),esc(row.checkpoint_hash?.slice(0,12)||"Not recorded")];});
  const headers=["Run","Target minutes","Actual training time","Evidence domain","Result","Episodes","Checkpoint"];
  const checkpoints=data.runs.flatMap(run=>(run.artifact_manifest||[]).filter(artifact=>artifact.kind==="checkpoint"&&finite(artifact.metadata?.target_seconds)).map(artifact=>({run,artifact})));
  return panel("Governed checkpoint evaluations", "Completed evaluations of cold-start runs with a declared 3,600-second budget. Badges use requested checkpoint targets; actual snapshot times remain separate.",rail+`<div class="notice">A saved checkpoint establishes that a model was captured. Qualification requires measured held-out behavior in the actual game under a declared competence threshold.</div>`+(records.length?table(headers,evaluationRows(records)):empty("One-hour outcome not measured","No complete real-game qualification can be inferred from synthetic or uncalibrated simulator runs.")))+(continued.length?`<div style="margin-top:20px">${panel("Continued or other checkpoint evaluations","These records do not populate the cold-start one-hour badges. Times are local to the training run and exclude inherited training.",table(["Training class",...headers],evaluationRows(continued).map((row,index)=>[esc(trainingClass(continued[index])),...row])))}</div>`:"")+`<div class="grid" style="margin-top:20px">${panel("Recorded checkpoint artifacts","Training class and local target/actual times preserve the distinction between cold starts and continued training.",table(["Run","Training class","Target time","Actual training time","SHA-256"],checkpoints.map(({run,artifact})=>[runButton(run),esc(run.metadata?.benchmark_class||"Not recorded"),seconds(artifact.metadata.target_seconds),seconds(artifact.metadata.training_elapsed_seconds),esc(artifact.sha256.slice(0,20))])))}</div>`;
}
function transfer(data) {
  const generalization=data.evaluations.filter(row=>["in_distribution","new_map","new_vehicle","new_vehicle_and_map"].includes(row.metadata?.condition)||row.protocol?.includes("generalization"));
  const rows=generalization.map(row=>{const run=byId(row.run_id),[label,value]=quality(row);return [run?runButton(run):esc(row.run_id.slice(0,8)),`${esc(row.metadata?.condition||"held-out")}<span class="model-qualifier">${esc(evaluationContext(row))}</span>`,esc(conditionName(trainedCondition(row))),esc(conditionName(evaluatedCondition(row))),`${num(value,2)}<span class="model-qualifier">${esc(label)}</span>`,esc(row.results?.scope||domain(run))];});
  const adaptation=data.evaluations.filter(row=>finite(adaptationMinutes(row)));
  const adaptationGroups=new Map();for(const row of adaptation){const run=trainingRun(row)||byId(row.run_id),key=`${run?runName(run):row.run_id} · ${run?.run_id.slice(0,8)||row.run_id.slice(0,8)} · ${conditionName(evaluatedCondition(row))}`;if(!adaptationGroups.has(key))adaptationGroups.set(key,[]);adaptationGroups.get(key).push({x:adaptationMinutes(row),y:quality(row)[1]});}
  const adaptedSeries=[...adaptationGroups].map(([name,points],index)=>({name,points,color:palette[index%palette.length]}));
  const paired=new Map();for(const row of data.evaluations){if(!row.checkpoint_hash)continue;const key=`${row.checkpoint_hash}|${row.metadata?.comparison_condition || ""}`;if(!paired.has(key))paired.set(key,[]);paired.get(key).push(row);}
  const gaps=[];for(const [key,items]of paired){const real=items.find(row=>["real_game","real-game"].includes(domain(byId(row.run_id)))),sim=items.find(row=>["simulation","uncalibrated simulation"].includes(domain(byId(row.run_id))));if(real&&sim){const [realName,realValue]=quality(real),[simName,simValue]=quality(sim);if(realName===simName&&finite(realValue)&&finite(simValue))gaps.push([esc(key.split("|")[0].slice(0,14)),num(realValue,2),num(simValue,2),simValue!==0?num(realValue/simValue,3):"Undefined (zero simulator score)",esc(realName)]);}}
  return `<div class="grid">${panel("Vehicle × map generalization", "Original condition labels describe the source evaluation grid. Latest training conditions come from the saved policy/configuration; an adapted condition is not newly held out.",rows.length?table(["Run","Source condition label","Latest trained vehicle / map","Evaluation vehicle / map","Result","Scope"],rows):empty("Generalization not measured","Evaluate held-out maps, vehicles, and their combined change with explicit condition labels."))}</div><div class="grid equal">${panel("Adaptation over exposure", "Recorded quality versus actual additional training minutes in a changed vehicle/map condition. Parent training is excluded; same-condition continuation is excluded.",chart(adaptedSeries,{xLabel:"Additional adaptation minutes"}))}${panel("Sim-to-real transfer", "Ratios require matching checkpoint hashes, condition labels and measurement definitions.",gaps.length?table(["Checkpoint","Real","Simulator","Real / sim","Metric"],gaps):empty("Transfer gap not measured","No matched real-game and simulator evaluation pair is available."))}</div>`;
}
function controls(data) {
  const runSelect=`<div class="local-controls"><label>Trajectory run<select id="trajectory-run"><option value="">Most recent with trajectories</option>${data.runs.filter(run=>data.controls.some(row=>row.run_id===run.run_id)).map(run=>`<option value="${esc(run.run_id)}">${esc(runName(run))}</option>`).join("")}</select></label></div>`;
  const selected=state.trajectoryRun||data.controls[0]?.run_id;
  const rows=data.controls.filter(row=>row.run_id===selected);
  const trace=[{name:"Requested gas",color:palette[0],points:rows.map(row=>({x:row.elapsed_seconds,y:row.gas?1:0}))},{name:"Requested brake / reverse",color:palette[2],points:rows.map(row=>({x:row.elapsed_seconds,y:row.brake?1:0}))}];
  const traces=rows.length?runSelect+`<div class="notice">These are recorded command states. Leases can overlap or end early; actual OS contact/key durations require the saved input-transition artifacts. Commands do not establish game acknowledgment.</div>`+chart(trace,{zero:true,stepped:true,yLabel:"Requested state"})+table(["Episode","Step","Elapsed","Gas","Brake","Recorded duration"],rows.slice(0,100).map(row=>[esc(row.episode_id),esc(row.step),seconds(row.elapsed_seconds),row.gas?"1":"0",row.brake?"1":"0",seconds(row.action_duration_seconds)])):empty("Temporal pedal commands not measured","All four states (00, 10, 01, 11) are represented in the schema. Episode action counts alone do not reconstruct action duration.");
  const counts=data.evaluations.filter(row=>Array.isArray(row.results?.action_counts));
  const actionTable=counts.length?table(["Run","00 · neither","10 · gas","01 · brake","11 · both"],counts.map(row=>{const run=byId(row.run_id);return [run?runButton(run):esc(row.run_id.slice(0,8)),...row.results.action_counts.map(value=>compact(value))];})):empty("Action frequencies not measured","Independent gas and brake channels can be analyzed when a policy evaluation records its actions.");
  const artifacts=data.runs.flatMap(run=>(run.artifact_manifest||[]).filter(item=>item.kind.includes("calibration")).map(artifact=>({run,artifact})));
  const calibrationMetrics=data.metrics.filter(row=>row.name.includes("calibration")||row.name.includes("trajectory_error"));
  const calibration=calibrationMetrics.length?table(["Run","Error measurement","Value","Step"],calibrationMetrics.map(row=>[byId(row.run_id)?runButton(byId(row.run_id)):esc(row.run_id),esc(metricLabel(row.name)),num(row.value,4),esc(row.step)])):empty("Simulator calibration not measured","A simulator with no held-out real trajectory error remains an uncalibrated surrogate.");
  return `<div class="grid">${panel("Recorded pedal commands","Gas and brake are independent. Hover a sample for its requested state and dispatch timestamp.",traces,state.controlsTruncated?"First 10,000 trajectory rows":"Canonical trajectories")}${panel("Observed pedal-state distribution","Counts are taken from complete evaluation records. They are not a scripted control policy.",actionTable)}${panel("Calibration and held-out trajectory error","Source trajectories, fitted parameters and validation errors must be linked through a calibration record.",calibration+ (artifacts.length?table(["Run","Artifact","Hash"],artifacts.map(({run,artifact})=>[runButton(run),esc(artifact.path),esc(artifact.sha256.slice(0,16))])):""))}</div>`;
}
const efficiencyCostLabels = {
  cpu_core_seconds:"CPU core-seconds",gpu_utilization_equivalent_seconds:"GPU utilization-equivalent seconds",
  simulator_transitions:"Simulator transitions",simulator_episodes:"Simulator episodes",simulator_seconds:"Simulated seconds",
  physics_steps:"Physics steps",rendered_frames:"Rendered frames",policy_decisions:"Policy decisions",optimizer_updates:"Optimizer updates",
  real_game_interaction_seconds:"Real-game interaction seconds",real_game_episodes:"Real-game episodes",
  data_generation_cpu_core_seconds:"Data-generation CPU core-seconds",data_generation_gpu_utilization_equivalent_seconds:"Data-generation GPU utilization-equivalent seconds"
};
const efficiencyStatus = {reached:"Reached",right_censored:"Right censored",not_evaluable:"Not evaluable"};
const efficiencySeconds = value => finite(value) ? `${num(value,6)} s` : "Not measured";
function efficiencyCostTable(point) {
  return table(["Resource or experience","Own","Prior","Combined"],Object.entries(efficiencyCostLabels).map(([key,label])=>[esc(label),...['own_cost','prior_cost','combined_cost'].map(scope=>num(point[scope]?.[key],6))]));
}
function efficiencySelection() { return state.efficiency.filter(report=>!state.efficiencyStudy||report.study_id===state.efficiencyStudy); }
function realEfficiency() {
  if(!state.efficiencyLoaded)return panel("Reading efficiency evidence","Waiting for the canonical study snapshot.",empty("Evidence loading","An empty result has not yet been established."));
  if(state.efficiencyError)return panel("Efficiency evidence unavailable","Historical views remain independently available.",empty("Could not validate efficiency evidence",state.efficiencyError));
  const reports=efficiencySelection();
  const select=`<div class="local-controls"><label>Study<select id="efficiency-study"><option value="">All registered studies</option>${state.efficiency.map(report=>`<option value="${esc(report.study_id)}"${state.efficiencyStudy===report.study_id?' selected':''}>${esc(report.study_id)}</option>`).join('')}</select></label></div>`;
  const scopeNote=`<div class="notice">This view uses each study's frozen population. The study selector changes displayed records; it does not recompute outcomes or the registered comparison cohorts. Episodes measure experience, compute measures cost, and elapsed time measures rapidity. Missing values remain unknown.</div>`;
  if(!state.efficiency.length)return panel("Real competence efficiency is not yet measured","Validated Cycle 3 study envelopes are required.",empty("No validated efficiency studies","Historical simulation curves do not establish real-game competence. They remain available in Learning & efficiency."));
  const errors=reports.filter(report=>report.error).map(report=>panel(`Evidence validation failed · ${report.study_id}`,"No competence or efficiency claim is available for this record.",`<div class="notice">${esc(report.error)}</div>`)).join('');
  const studies=reports.filter(report=>!report.error);
  const rows=studies.flatMap(report=>report.thresholds.map(point=>[
    `${esc(report.study_id)}<span class="model-qualifier">${esc(report.profile_id)} · ${esc(report.prior_class)} · ${esc(report.protocol_id)}</span>`,
    `${num(point.threshold_m)} m`,esc(efficiencyStatus[point.status]||point.status),efficiencySeconds(point.time_seconds),
    efficiencySeconds(point.interval_lower_seconds),efficiencySeconds(point.censor_seconds),efficiencySeconds(point.budget_seconds),
    esc(point.checkpoint_sha256||"Not measured")
  ]));
  const thresholdTable=panel("Time to real competence","The first observed passing frozen checkpoint bounds acquisition time. Censor support ends at the last eligible evaluation, which can precede the declared budget.",rows.length?table(["Study / frozen cohort","Threshold","Status","Reached at · own elapsed","Earlier evaluation","Censor support","Declared budget","Checkpoint SHA-256"],rows):empty("Thresholds not evaluable","The visible source errors must be resolved before evaluating these studies."));
  const series=studies.slice(0,12).map((report,index)=>({name:`${report.study_id} · ${report.profile_id} · ${report.prior_class}`,color:palette[index%palette.length],points:(report.checkpoint_curve||[]).map(point=>({x:point.time_seconds,y:point.median_distance_m}))}));
  const curve=panel("Independently evaluated checkpoints","Dots show recorded median real-game distance at each eligible frozen checkpoint. No interpolation or monotonic learning is assumed; protocol and profile define comparability.",chart(series,{scatter:true,xLabel:"Own command-start elapsed seconds",yLabel:"Median real-game distance (m)"}),studies.length>12?"First 12 studies plotted · export includes all":"Canonical checkpoint observations");
  const detail=studies.map(report=>{
    const warnings=(report.warnings||[]).map(warning=>`<li>${esc(warning)}</li>`).join('');
    const identity=table(["Definition","Recorded value"],[
      ["Prior class",esc(report.prior_class)],["Profile / protocol",`${esc(report.profile_id)} / ${esc(report.protocol_id)}`],
      ["Real scenario SHA-256",esc(report.scenario_sha256||"Not measured")],["Comparison contract SHA-256",esc(report.comparison_contract_sha256||"Not measured")],
      ["Evaluation split",esc(report.evaluation_split)],["Compute comparison scope",esc(report.compute_comparison_scope)],
      ["Wall-clock boundary",esc(report.wall_clock_boundary)],["Lineage complete",report.lineage_complete?"Yes":"No"],
      ["Provenance",esc(report.provenance_status||"Not measured")]
    ]);
    const clocks=table(["Checkpoint","Own elapsed","Verification elapsed","Evaluation elapsed","Evaluation run"],(report.checkpoint_curve||[]).map(point=>[esc(point.checkpoint_sha256),efficiencySeconds(point.time_seconds),efficiencySeconds(point.verification_elapsed_seconds),efficiencySeconds(point.evaluation_elapsed_seconds),esc(point.evaluation_run_id)]));
    const costs=report.thresholds.map(point=>`<details><summary>${num(point.threshold_m)} m · ${esc(efficiencyStatus[point.status]||point.status)} · own / prior / combined costs</summary>${efficiencyCostTable(point)}</details>`).join('');
    const prior=table(["Cost ID","Kind","Parent IDs","Separate prior elapsed","Human time"],(report.prior_ledger||[]).map(item=>[esc(item.cost_id),esc(item.kind),esc((item.parents||[]).join(', ')||"None"),efficiencySeconds(item.elapsed_seconds),efficiencySeconds(item.human_seconds)]));
    const evaluation=table(["Resource or experience","Evaluation cost"],Object.entries(efficiencyCostLabels).map(([key,label])=>[esc(label),num(report.evaluation_cost?.[key],6)]));
    return panel(report.study_id,"Costs are reported at the selected checkpoint or censor support. Data-generation compute is a subset of total compute. Evaluation cost is separate.",identity+(warnings?`<ul class="notice">${warnings}</ul>`:'')+`<details><summary>Checkpoint evaluation and verification clocks</summary>${clocks}</details>`+costs+`<details><summary>Prior elapsed clocks and ancestry</summary><p>Policy prior roots: ${pretty(report.policy_prior_roots||[])}. Parallel or shared prior elapsed clocks are never added to the own elapsed headline. Each listed ancestor is charged once by the canonical analysis.</p>${(report.prior_ledger||[]).length?prior:'<p>No inherited cost nodes recorded. The lineage declaration above remains authoritative.</p>'}</details><details><summary>Evaluation costs (separate)</summary>${evaluation}</details><details><summary>Complete canonical analysis and provenance</summary><pre>${pretty(report)}</pre></details>`);
  }).join('');
  const comparisons=state.efficiencyComparisons.map(comparison=>{
    const included=(comparison.comparisons||[]).filter(item=>!state.efficiencyStudy||item.study_id===state.efficiencyStudy);
    const excluded=(comparison.excluded||[]).filter(item=>!state.efficiencyStudy||item.study_id===state.efficiencyStudy);
    const comparisonRows=included.map(item=>[esc(item.study_id),item.pareto_frontier?"Pareto frontier":"Dominated",esc(item.cohort.join(' / ')),pretty(item.values),esc(item.dominated_by.join(', ')||"None")]).concat(excluded.map(item=>[esc(item.study_id),"Excluded",esc(item.reason),"Not measured","Not measured"]));
    return panel(`Pareto comparison · ${num(comparison.threshold_m)} m`,comparison.note,table(["Study","Comparison status","Frozen cohort / exclusion reason","Measured dimensions","Dominated by"],comparisonRows));
  }).join('');
  return select+scopeNote+`<div class="grid">${errors}${thresholdTable}${curve}${detail}${comparisons}</div>`;
}
function render() {
  const data=scope(), [title,description,crumb]=titles[state.view];
  $("#page-title").textContent=title;$("#page-description").textContent=description;$("#breadcrumb").textContent=crumb;
  document.querySelectorAll(".nav-button").forEach(button=>button.classList.toggle("active",button.dataset.view===state.view));
  $("#content").innerHTML=({overview,runs:runsView,learning,"real-efficiency":realEfficiency,benchmark,transfer,controls}[state.view])(data);
  $(".filters").hidden=state.view==="real-efficiency";
  $("#export").disabled=state.view==="real-efficiency"&&!state.efficiencyLoaded;
  $("#source-status").textContent=state.view==="real-efficiency"?"Validated Cycle 3 study envelopes · canonical efficiency analysis":"Canonical experiment records · DuckDB / Parquet";
  $("#scope-status").textContent=state.view==="real-efficiency"?`${efficiencySelection().length} of ${state.efficiency.length} study envelopes`:`${data.runs.length} of ${state.runs.length} runs · ${data.metrics.length.toLocaleString()} metric observations`;
  $("#efficiency-study")?.addEventListener("change",event=>{state.efficiencyStudy=event.target.value;render();});
  $("#metric")?.addEventListener("change",event=>{state.metric=event.target.value;render();});
  $("#axis")?.addEventListener("change",event=>{state.axis=event.target.value;render();});
  $("#threshold")?.addEventListener("change",event=>{state.threshold=event.target.value;render();});
  $("#search")?.addEventListener("change",event=>{state.search=event.target.value;render();});
  $("#simulator")?.addEventListener("change",event=>{state.simulator=event.target.value;render();});
  $("#after")?.addEventListener("change",event=>{state.after=event.target.value;render();});
  $("#minimum-duration")?.addEventListener("change",event=>{state.minimumDuration=Math.max(0,Number(event.target.value));render();});
  $("#sort")?.addEventListener("change",event=>{state.sort=event.target.value;render();});
  $("#trajectory-run")?.addEventListener("change",event=>{state.trajectoryRun=event.target.value;render();});
  if($("#trajectory-run")&&state.trajectoryRun)$("#trajectory-run").value=state.trajectoryRun;
  document.querySelectorAll("[data-run]").forEach(button=>button.addEventListener("click",()=>openDetail(button.dataset.run)));
}
async function request(path) { const response=await fetch(path,{cache:"no-store"}); if(!response.ok)throw Error(`${response.status}: could not read ${path}`); return response.json(); }
async function refresh() {
  $("#refresh").disabled=true;$("#error").classList.add("hidden");
  const efficiencyRefresh=request("/api/learning-efficiency/snapshot").then(snapshot=>{
    Object.assign(state,{efficiency:snapshot.studies,efficiencyComparisons:snapshot.comparisons,efficiencyError:"",efficiencyLoaded:true});
    if(!state.efficiency.some(report=>report.study_id===state.efficiencyStudy))state.efficiencyStudy="";
  }).catch(error=>{Object.assign(state,{efficiency:[],efficiencyComparisons:[],efficiencyError:error.message,efficiencyLoaded:true});}).then(()=>{if(state.view==="real-efficiency")render();});
  try {
    const [runs,metrics,evaluations,resources,controls]=await Promise.all([request("/api/runs"),request("/api/metrics"),request("/api/analytics/evaluations"),request("/api/analytics/resources"),request("/api/analytics/controls")]);
    Object.assign(state,{runs,metrics,evaluations:evaluations.rows,resources:resources.rows,controls:controls.rows,resourcesTruncated:resources.truncated,controlsTruncated:controls.truncated});
    for(const [id,key,label]of [["algorithm","algorithm","algorithms"],["environment","environment","environments"],["status","status","statuses"],["vehicle","vehicle_profile","vehicles"],["map","map_profile","maps"]]){
      const select=$("#"+id),value=select.value,options=[...new Set(runs.map(run=>run[key]).filter(Boolean))].sort();
      select.innerHTML=`<option value="">All ${label}</option>`+options.map(option=>`<option value="${esc(option)}">${esc(option)}</option>`).join("");
      if(options.includes(value))select.value=value;
    }
    $("#loading").classList.add("hidden");$("#updated").textContent=`Read ${new Date().toLocaleTimeString(undefined,{hour:"2-digit",minute:"2-digit"})}`;render();
  }catch(error){$("#loading").classList.add("hidden");$("#error").textContent=`The experiment store could not be read. ${error.message}`;$("#error").classList.remove("hidden");}
  finally{await efficiencyRefresh;$("#refresh").disabled=false;}
}
async function openDetail(id) {
  const run=byId(id);if(!run)return;
  $("#detail-title").textContent=runName(run);
  const fields={"Run ID":run.run_id,"Experiment":run.experiment_id,"Status":run.status,"Source SHA":run.git_sha||"Unavailable","Worktree":run.dirty_worktree===true?"Dirty at start":run.dirty_worktree===false?"Clean at start":"Unavailable","Project version":run.project_version,"Started":date(run.start_time),"Duration":seconds(run.duration),"Seed":run.seed,"Environment":run.environment,"Vehicle / map":`${run.vehicle_profile||"Not recorded"} / ${run.map_profile||"Not recorded"}`,"Configuration hash":run.config_sha256};
  $("#detail-body").innerHTML=`<div class="detail-grid">${Object.entries(fields).map(([name,value])=>`<div class="detail-item">${esc(name)}<strong>${esc(value)}</strong></div>`).join("")}</div><div id="integrity-status" class="integrity">Checking the immutable record seal…</div><details open><summary>Configuration</summary><pre>${pretty(run.configuration)}</pre></details><details><summary>Machine and framework versions</summary><pre>${pretty({machine_fingerprint:run.machine_fingerprint,hardware:run.hardware,cpu:run.cpu,gpu:run.gpu,ram:run.ram,driver:run.driver,cuda:run.cuda,python:run.python_version,frameworks:run.framework_versions})}</pre></details><details><summary>Evaluation and summary</summary><pre>${pretty({evaluations:run.evaluation_results,summary:run.summary})}</pre></details><details><summary>Artifact manifest (${run.artifact_manifest?.length||0})</summary><pre>${pretty(run.artifact_manifest)}</pre></details><details><summary>Complete scientific record</summary><pre>${pretty(run)}</pre></details><button class="button primary" id="export-run">Export run JSON ↓</button>`;
  $("#export-run").addEventListener("click",()=>download(`gradientclimb-${id}.json`,run));
  $("#detail-dialog").showModal();
  try{const integrity=await request(`/api/runs/${id}/integrity`);if($("#integrity-status"))$("#integrity-status").textContent=integrity.valid?"✓ All sealed files and artifacts match their SHA-256 hashes.":`Seal: ${integrity.errors.join("; ")}`;}catch(error){if($("#integrity-status"))$("#integrity-status").textContent=`Integrity verification unavailable: ${error.message}`;}
}
function download(name,payload){const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"}),link=document.createElement("a");link.href=URL.createObjectURL(blob);link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);}
$("#refresh").addEventListener("click",refresh);
$("#close-detail").addEventListener("click",()=>$("#detail-dialog").close());
document.querySelectorAll(".nav-button").forEach(button=>button.addEventListener("click",()=>{state.view=button.dataset.view;render();}));
document.querySelectorAll(".filters select").forEach(select=>select.addEventListener("change",render));
$("#reset").addEventListener("click",()=>{document.querySelectorAll(".filters select").forEach(select=>select.value="");state.search="";state.simulator="";state.minimumDuration=0;state.after="";render();});
$("#export").addEventListener("click",()=>{if(state.view==="real-efficiency"){download("gradientclimb-real-efficiency.json",{source:"Canonical Cycle 3 efficiency analysis",selected_study:state.efficiencyStudy||null,studies:efficiencySelection(),comparisons:state.efficiencyComparisons,error:state.efficiencyError||null});return;}const data=scope();download(`gradientclimb-${state.view}.json`,{exported_at:new Date().toISOString(),source:"Canonical run records and Parquet through DuckDB",view:state.view,selected_metric:state.metric,horizontal_axis:state.axis,filters:{algorithm:$("#algorithm").value,environment:$("#environment").value,status:$("#status").value,vehicle:$("#vehicle").value,map:$("#map").value,search:state.search,simulator:state.simulator,after:state.after,minimum_duration:state.minimumDuration},...data,metric_rows_for_selected_measurement:data.metrics.filter(row=>row.name===state.metric),resource_rows_truncated:state.resourcesTruncated,trajectory_rows_truncated:state.controlsTruncated});});
refresh();
