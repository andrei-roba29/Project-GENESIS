/* Project GENESIS — Phase 1 frontend application.
 * Vanilla JS, zero dependencies, works fully offline against the local server.
 */
(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => [...document.querySelectorAll(s)];

  let state = null;          // last /api/state payload
  let layer = "elevation";
  let viewMode = "globe";
  let activeTab = "world";
  let lastVersion = -1;
  let textureImg = null;
  let selectedCell = null;
  let globeReady = false;

  /* ---------------- API helpers ---------------- */
  async function api(path, opts = {}) {
    const res = await fetch(path, {
      method: opts.method || "GET",
      headers: { "Content-Type": "application/json" },
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  }

  /* ---------------- globe / map ---------------- */
  const globe = new GenesisGlobe.Globe($("#globe"), (lat, lon) => inspect(lat, lon));
  const flatMap = new GenesisGlobe.FlatMap($("#map2d"), (lat, lon) => inspect(lat, lon));
  const ground = new GenesisGlobe.GroundView($("#ground"), (g) => {
    $("#ground-coord").textContent =
      `${g.lat.toFixed(1)}°, ${g.lon.toFixed(1)}°`;
  });
  // Report an unrenderable globe instead of showing an empty void.
  if (globe.error) showGlobeError(globe.error);
  if (!globe.prog) showGlobeError("WebGL is unavailable in this browser");

  function showGlobeError(msg) {
    const el = $("#viewer-error");
    if (el) { el.textContent = "3D view unavailable: " + msg; el.style.display = "block"; }
  }

  /* Deep zoom hands over to the 3D ground view automatically: the globe view
   * is for large-scale reading, the ground view is for standing in the world. */
  function maybeEnterGroundView() {
    if (viewMode !== "globe" || !globeReady) return;
    if (globe.zoom <= 1.16) enterGroundView();
  }

  function enterGroundView() {
    const lat = selectedCell ? selectedCell.lat : globe.pickLat;
    const lon = selectedCell ? selectedCell.lon : globe.pickLon;
    // Rotate the globe so the point being inspected is centred, so the dive
    // lands where the user is actually looking.
    if (viewMode === "globe" && globe.focusOn) globe.focusOn(lat, lon);
    viewMode = "ground";
    $("#globe").style.display = "none";
    $("#map2d").style.display = "none";
    $("#ground").style.display = "block";
    $("#ground-hud").style.display = "flex";
    $("#viewport-hint").style.display = "none";
    $("#btn-view").textContent = "🌐 Globe";
    ground.load(lat, lon, parseFloat($("#ground-span").value), layer);
  }

  function exitGroundView() {
    viewMode = "globe";
    $("#ground").style.display = "none";
    $("#ground-hud").style.display = "none";
    $("#globe").style.display = "block";
    $("#viewport-hint").style.display = "block";
    $("#btn-view").textContent = "🗺 Map";
    globe.zoom = 1.45;
  }

  $("#ground-back").onclick = exitGroundView;
  $("#ground-exag").oninput = (e) => {
    const ex = parseFloat(e.target.value);
    $("#ground-exag-val").textContent = ex + "×";
    ground.exaggeration = ex;
    // Rebuild from the cached heightfield: no server round-trip needed.
    if (ground._lastMesh) ground._buildMesh(ground._lastMesh, ground._lastSize);
  };
  $("#ground-veg").onchange = (e) => { ground.showVegetation = e.target.checked; };
  $("#ground-clouds").onchange = (e) => { ground.showClouds = e.target.checked; };
  $("#ground-span").oninput = (e) => {
    const span = parseFloat(e.target.value);
    $("#ground-span-val").textContent = span + "°";
    ground.load(undefined, undefined, span, layer);
  };

  function loadTexture() {
    const img = new Image();
    img.onload = () => {
      textureImg = img;
      globe.setTextureImage(img);
      flatMap.setImage(img);
      globeReady = true;
    };
    img.src = `/api/texture?layer=${layer}&v=${lastVersion}`;
  }

  function frame() {
    if (viewMode === "globe") { globe.draw(); maybeEnterGroundView(); }
    else if (viewMode === "map") flatMap.draw();
    else ground.draw();
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  /* ---------------- polling ---------------- */
  async function poll() {
    try {
      const s = await api("/api/state");
      if (!s.generated) { $("#loading-overlay").style.display = "flex"; return; }
      $("#loading-overlay").style.display = "none";
      state = s;
      const sum = s.summary;
      $("#date").textContent = sum.date;
      $("#season").textContent = sum.season;
      $("#north-season").innerHTML = `N: <b>${sum.season_north}</b>`;
      $("#south-season").innerHTML = `S: <b>${sum.season_south}</b>`;
      const meta = $("#ground-meta");
      if (meta && viewMode === "ground") {
        const lat = ground.lat;
        const local = lat >= 0 ? sum.season_north : sum.season_south;
        meta.textContent = `${local} · sun ${sum.declination_deg > 0 ? "N" : "S"}${Math.abs(sum.declination_deg).toFixed(0)}°`;
      }
      $("#btn-pause").textContent = sum.paused ? "▶ Resume" : "⏸ Pause";
      $$(".speed-group button").forEach(b =>
        b.classList.toggle("active", parseFloat(b.dataset.speed) === sum.speed_multiplier));
      $("#stat-date").innerHTML = `Date <b>${sum.date}</b>`;
      $("#stat-ocean").innerHTML = `Ocean <b>${(sum.ocean_fraction * 100).toFixed(1)}%</b>`;
      $("#stat-temp").innerHTML = `Mean temp <b>${sum.mean_temp_c}°C</b>`;
      $("#stat-veg").innerHTML = `Vegetation <b>${(sum.mean_veg_cover * 100).toFixed(0)}%</b>`;
      $("#stat-species").innerHTML = `Species <b>${sum.species_in_radius}</b>`;
      $("#stat-health").innerHTML = `Health <b>${(sum.health * 100).toFixed(0)}%</b>`;
      $("#stat-radius").innerHTML = `Fauna radius <b>${sum.fauna_radius_km} km</b>`;
      $("#stat-traversal").innerHTML = `Crossing <b>${sum.traversal_minutes} min</b>`;
      if (sum.version !== lastVersion) {
        lastVersion = sum.version;
        loadTexture();
        refreshTab();
      }
      if (selectedCell) inspect(selectedCell.lat, selectedCell.lon, true);
    } catch (e) { console.warn("poll failed", e); }
  }
  setInterval(poll, 1000);

  /* ---------------- header controls ---------------- */
  $("#btn-pause").onclick = async () => {
    await api("/api/control", { method: "POST", body: { paused: !state.summary.paused } });
    poll();
  };
  $$(".speed-group button").forEach(b => b.onclick = async () => {
    await api("/api/control", { method: "POST", body: { speed_multiplier: parseFloat(b.dataset.speed) } });
    poll();
  });
  $("#btn-tick").onclick = async () => { await api("/api/tick", { method: "POST", body: { days: 30 } }); poll(); };
  $("#layer-select").onchange = (e) => {
    layer = e.target.value;
    loadTexture();
    if (viewMode === "ground") { ground._loadedKey = null; ground.load(undefined, undefined, undefined, layer); }
  };
  $("#btn-view").onclick = () => {
    if (viewMode === "ground") { exitGroundView(); return; }
    if (viewMode === "globe") {
      viewMode = "map";
      $("#globe").style.display = "none";
      $("#map2d").style.display = "block";
      $("#btn-view").textContent = "🌐 Globe";
      flatMap.fit();
    } else {
      exitGroundView();
    }
  };

  /* ---------------- tabs ---------------- */
  $$("#tabs button").forEach(b => b.onclick = () => {
    activeTab = b.dataset.tab;
    $$("#tabs button").forEach(x => x.classList.toggle("active", x === b));
    refreshTab();
  });

  async function refreshTab() {
    const el = $("#tab-content");
    if (!state) return;
    const sum = state.summary;
    try {
      if (activeTab === "world") {
        const hist = await api("/api/history");
        el.innerHTML = `
          <h2>World</h2>
          <div class="kv"><span>Name</span><span>${sum.name}</span></div>
          <div class="kv"><span>Seed</span><span>${sum.seed}</span></div>
          <div class="kv"><span>Preset</span><span>${sum.preset}</span></div>
          <div class="kv"><span>Resolution</span><span>${sum.width}×${sum.height}</span></div>
          <div class="kv"><span>Radius</span><span>${sum.radius_km} km</span></div>
          <div class="kv"><span>Circumference</span><span>${sum.circumference_km} km</span></div>
          <div class="kv"><span>Crossing time</span><span>${sum.traversal_minutes} min @ ${sum.movement_speed_kmh} km/h</span></div>
          <div class="kv"><span>Ocean</span><span>${(sum.ocean_fraction * 100).toFixed(1)}%</span></div>
          <div class="kv"><span>Mean elevation</span><span>${sum.mean_elevation_m} m</span></div>
          <div class="kv"><span>Highest point</span><span>${sum.max_elevation_m} m</span></div>
          <div class="kv"><span>Deepest point</span><span>${sum.min_elevation_m} m</span></div>
          <div class="kv"><span>Simulated date</span><span>${sum.date} (${sum.season})</span></div>
          <div class="kv"><span>Clock rate</span><span>${sum.days_per_real_second} days/sec (100 yr/hr)</span></div>
          <h3>Ecosystem health over time</h3>
          <canvas class="chart" id="chart-health"></canvas>
          <h3>Vegetation cover over time</h3>
          <canvas class="chart" id="chart-veg"></canvas>`;
        drawChart($("#chart-health"), hist.stats, "health", "#37d99a");
        drawChart($("#chart-veg"), hist.stats, "veg_cover", "#4da3ff");
      } else if (activeTab === "geography") {
        const g = await api("/api/geography");
        el.innerHTML = `
          <h2>Geography</h2>
          <div class="kv"><span>Mean elevation</span><span>${g.mean_elevation_m} m</span></div>
          <div class="kv"><span>Highest</span><span>${g.max_elevation_m} m</span></div>
          <div class="kv"><span>Lowest</span><span>${g.min_elevation_m} m</span></div>
          <div class="kv"><span>Mean slope</span><span>${g.mean_slope}</span></div>
          <div class="kv"><span>Tectonic plates</span><span>${g.plate_count}</span></div>
          <div class="kv"><span>Mountain cells (&gt;2500 m)</span><span>${g.mountain_cells}</span></div>
          <div class="kv"><span>Lowland cells (&lt;300 m)</span><span>${g.lowland_cells}</span></div>
          <h3>Highest peaks</h3>
          <table><tr><th>Lat</th><th>Lon</th><th>Elevation</th><th>Biome</th></tr>
          ${g.peaks.map(p => `<tr><td>${p.lat}</td><td>${p.lon}</td><td>${p.elevation_m} m</td><td>${p.biome}</td></tr>`).join("")}</table>
          <p class="muted">Mountains form at convergent tectonic boundaries; rifts at divergent ones. Switch to the Elevation layer to see them.</p>`;
      } else if (activeTab === "climate") {
        el.innerHTML = `
          <h2>Climate</h2>
          <div class="kv"><span>Mean temperature</span><span>${sum.mean_temp_c}°C</span></div>
          <div class="kv"><span>Mean precipitation</span><span>${sum.mean_precip_mm} mm/yr</span></div>
          <div class="kv"><span>Northern hemisphere</span><span>${sum.season_north}</span></div>
          <div class="kv"><span>Southern hemisphere</span><span>${sum.season_south}</span></div>
          <div class="kv"><span>Day of year</span><span>${sum.day_of_year}</span></div>
          <div class="kv"><span>Solar declination</span><span>${sum.declination_deg}°</span></div>
          <div class="kv"><span>Mean seasonal swing</span><span>±${(sum.seasonal_amp_c / 2).toFixed(1)} °C</span></div>
          <h3>How the seasons work</h3>
          <p class="muted">The sub-solar point migrates between the tropics, so the two hemispheres are always in opposite seasons. Each cell's swing is scaled by latitude (the tropics barely move, mid-latitudes swing hardest), by continentality (inland areas swing more than coasts — measured on this planet's own scale), and by elevation (thinner air swings harder).</p>
          <p class="muted">Precipitation belts migrate with the sun too: the ITCZ and the westerlies shift toward the summer hemisphere, which is why wet seasons follow the warm season.</p>
          <p class="muted">Switch to the <b>Temperature</b> layer and let the clock run — the colour bands visibly migrate north and south with the seasons.</p>
          <h3>Temperature layer</h3>
          <p class="muted">Select “Temperature” in the layer menu to visualise the current seasonal field.</p>`;
      } else if (activeTab === "weather") {
        const hist = await api("/api/history");
        const latest = hist.stats[hist.stats.length - 1];
        const w = latest ? latest.weather : {};
        el.innerHTML = `
          <h2>Weather</h2>
          <div class="kv"><span>Snapshot</span><span>${latest ? latest.date : "—"}</span></div>
          ${Object.keys(w).length ? Object.entries(w).map(([k, v]) => `<div class="kv"><span>${k}</span><span>${v} cells</span></div>`).join("") : "<p class='muted'>No active phenomena in the last snapshot.</p>"}
          <h3>Recent events</h3>
          ${hist.history.filter(h => h.kind !== "world").slice(-12).reverse().map(h =>
            `<div class="event-item"><div>${h.text}</div><div class="when">${h.date}</div></div>`).join("") || "<p class='muted'>No events yet. Let the simulation run.</p>"}
          <p class="muted">Weather has real consequences: drought stresses vegetation, heavy rain floods lowlands and deposits sediment, lightning ignites wildfires in dry forests.</p>`;
      } else if (activeTab === "hydrology") {
        el.innerHTML = `
          <h2>Hydrology</h2>
          <div class="kv"><span>River cells</span><span>${sum.river_cells}</span></div>
          <div class="kv"><span>Lake cells</span><span>${sum.lake_cells}</span></div>
          <div class="kv"><span>Wetland cells</span><span>${sum.wetland_cells}</span></div>
          <div class="kv"><span>Glacier cells</span><span>${sum.glacier_cells}</span></div>
          <div class="kv"><span>Ocean coverage</span><span>${(sum.ocean_fraction * 100).toFixed(1)}%</span></div>
          <p class="muted">Rivers are derived from D8 flow accumulation over the depression-filled terrain — they always run downhill to lakes or the sea. Lakes occupy filled depressions; wetlands fringe water on flat ground; glaciers sit high and cold.</p>
          <p class="muted">Select the “Water Systems” layer to see rivers (blue), lakes, wetlands and ice.</p>`;
      } else if (activeTab === "flora") {
        const f = await api("/api/flora");
        el.innerHTML = `
          <h2>Flora</h2>
          <div class="kv"><span>Mean cover</span><span>${(f.mean_cover * 100).toFixed(1)}%</span></div>
          <div class="kv"><span>Mean biomass</span><span>${f.mean_biomass_tha} t/ha</span></div>
          <h3>Cover by functional group</h3>
          ${Object.entries(f.groups).map(([g, v]) => `
            <div class="kv"><span>${g}</span><span>${(v * 100).toFixed(1)}%</span></div>
            <div class="bar"><div style="width:${(v * 100).toFixed(1)}%"></div></div>`).join("")}
          <h3>Species catalog</h3>
          <table><tr><th>Species</th><th>Group</th><th>Temp range</th><th>Rain (mm)</th><th>Growth</th></tr>
          ${f.species.map(s => `<tr><td>${s.name}</td><td>${s.group}</td><td>${s.t_min}…${s.t_max}°C</td><td>${s.p_min}…${s.p_max}</td><td>${s.growth}</td></tr>`).join("")}</table>
          <p class="muted">Vegetation is simulated as aggregated populations per cell: germination, logistic growth, water/nutrient consumption, seed dispersal, competition, mortality and decomposition. Pollinators and seed-dispersing animals boost reproduction.</p>`;
      } else if (activeTab === "fauna") {
        const f = await api("/api/fauna");
        el.innerHTML = `
          <h2>Fauna</h2>
          <div class="kv"><span>Human population (radius driver)</span><span>${f.human_population}</span></div>
          <div class="kv"><span>Active radius</span><span>${f.radius_km} km</span></div>
          <div class="kv"><span>Species in radius</span><span>${f.species_in_radius} / ${f.catalog_size}</span></div>
          <div class="kv"><span>Anchor</span><span>${f.anchor.lat}, ${f.anchor.lon}</span></div>
          <p class="muted">Formula: <code>${f.formula}</code><br>
          Rmin ${f.rmin_km} km · Rmax ${f.rmax_km} km · growth ${f.growth}.
          Phase 1 has no human entities yet — this parameter stands in for the future living population so the radius can be exercised. Adjust it in Settings.</p>
          <input type="range" id="fauna-pop" min="0" max="10000" value="${f.human_population}">
          <div class="btn-row"><button id="fauna-apply" class="btn-primary">Apply population</button></div>
          <h3>Species populations</h3>
          <table><tr><th>Species</th><th>Class</th><th>Diet</th><th>Population</th><th>In radius</th></tr>
          ${f.species.map(s => `<tr><td>${s.name}</td><td>${s.class}</td><td>${s.diet}</td><td>${s.population}</td><td>${s.in_radius}</td></tr>`).join("")}</table>
          <p class="muted">Populations are region-based (one number per species per region, representing many animals). They grow toward habitat carrying capacity, limited by food, water and predation — never spawned arbitrarily.</p>`;
        $("#fauna-apply").onclick = async () => {
          const n = parseInt($("#fauna-pop").value, 10);
          await api("/api/control", { method: "POST", body: { human_population: n } });
          poll(); refreshTab();
        };
      } else if (activeTab === "ecosystem") {
        const e = await api("/api/ecosystem");
        const h = e.health, n = e.nutrients;
        el.innerHTML = `
          <h2>Ecosystem</h2>
          <div class="kv"><span>Health index</span><span>${h.health}</span></div>
          <div class="bar"><div style="width:${(h.health * 100).toFixed(0)}%"></div></div>
          <div class="kv"><span>Vegetation</span><span>${h.vegetation}</span></div>
          <div class="kv"><span>Fertility</span><span>${h.fertility}</span></div>
          <div class="kv"><span>Moisture</span><span>${h.moisture}</span></div>
          <div class="kv"><span>Biodiversity</span><span>${h.biodiversity} (${h.species_present} species)</span></div>
          <div class="kv"><span>Disturbance</span><span>${h.disturbance}</span></div>
          <div class="kv"><span>Mean biomass</span><span>${h.mean_biomass_tha} t/ha</span></div>
          <h3>Nutrient cycle</h3>
          <div class="kv"><span>Organic matter</span><span>${n.mean_organic_matter}</span></div>
          <div class="kv"><span>Nutrients</span><span>${n.mean_nutrients}</span></div>
          <div class="kv"><span>Acidity (pH)</span><span>${n.mean_acidity_ph}</span></div>
          <div class="kv"><span>Soil depth</span><span>${n.mean_soil_depth_cm} cm</span></div>
          <div class="kv"><span>Erosion</span><span>${n.mean_erosion}</span></div>
          <p class="muted">Rain → soil moisture → vegetation → herbivores → predators → decomposition → nutrients → vegetation. Fire, flood and drought disturb the loop; the system recovers through germination and seed dispersal.</p>`;
      } else if (activeTab === "resources") {
        const r = await api("/api/resources");
        el.innerHTML = `
          <h2>Resources</h2>
          <table><tr><th>Resource</th><th>Mean abundance</th><th>Rich cells</th></tr>
          ${Object.entries(r.totals).map(([k, v]) => `<tr><td>${k}</td><td>${v.mean}</td><td>${v.cells_abundant}</td></tr>`).join("")}</table>
          <p class="muted">Freshwater follows rivers, lakes and glaciers; stone and ores concentrate in mountains and tectonic belts; timber tracks forest biomass; salt and sand mark arid basins and coasts. Select the “Resources” layer to visualise.</p>`;
      } else if (activeTab === "settings") {
        const cfg = await api("/api/settings");
        el.innerHTML = `
          <h2>Settings</h2>
          <label class="settings-label">World name</label>
          <input type="text" id="set-name" value="${cfg.world_name}">
          <label class="settings-label">Seed</label>
          <input type="number" id="set-seed" value="${cfg.seed}">
          <label class="settings-label">Preset</label>
          <select class="settings" id="set-preset">
            ${["earthlike", "archipelago", "desert", "frozen", "custom"].map(p => `<option ${p === cfg.preset ? "selected" : ""}>${p}</option>`).join("")}
          </select>
          <label class="settings-label">Resolution (width × height)</label>
          <input type="number" id="set-width" value="${cfg.width}">
          <input type="number" id="set-height" value="${cfg.height}">
          <label class="settings-label">Planet radius (km)</label>
          <input type="number" id="set-radius" value="${cfg.radius_km}">
          <label class="settings-label">Axial tilt (°)</label>
          <input type="number" id="set-tilt" value="${cfg.axial_tilt_deg}">
          <label class="settings-label">Ocean fraction</label>
          <input type="number" step="0.01" id="set-ocean" value="${cfg.ocean_fraction}">
          <label class="settings-label">Tectonic plates</label>
          <input type="number" id="set-plates" value="${cfg.plate_count}">
          <label class="settings-label">Movement speed (km/h) — crossing ≈ <span id="set-traversal">${(2 * Math.PI * cfg.radius_km / cfg.movement_speed_kmh * 60).toFixed(1)}</span> min</label>
          <input type="number" id="set-speed" value="${cfg.movement_speed_kmh}">
          <label class="settings-label">Human population (fauna radius driver)</label>
          <input type="number" id="set-humans" value="${cfg.human_population}">
          <div class="btn-row">
            <button id="set-generate" class="btn-primary">🌍 Generate new world</button>
            <button id="set-apply">Apply settings</button>
          </div>
          <h3>Persistence (local files, no cloud)</h3>
          <div class="btn-row">
            <button id="set-save">💾 Save world</button>
            <button id="set-export">⬇ Export JSON</button>
            <button id="set-import">⬆ Import JSON</button>
            <input type="file" id="import-file" accept=".json" style="display:none">
          </div>
          <p class="muted">Saves live in the <code>saves/</code> folder next to the server. Close the app and resume the same world later.</p>`;
        const readCfg = () => ({
          world_name: $("#set-name").value, seed: parseInt($("#set-seed").value, 10),
          preset: $("#set-preset").value, width: parseInt($("#set-width").value, 10),
          height: parseInt($("#set-height").value, 10), radius_km: parseFloat($("#set-radius").value),
          axial_tilt_deg: parseFloat($("#set-tilt").value), ocean_fraction: parseFloat($("#set-ocean").value),
          plate_count: parseInt($("#set-plates").value, 10),
          movement_speed_kmh: parseFloat($("#set-speed").value),
          human_population: parseInt($("#set-humans").value, 10),
        });
        $("#set-generate").onclick = async () => {
          $("#loading-overlay").style.display = "flex";
          await api("/api/generate", { method: "POST", body: { config: readCfg() } });
          lastVersion = -1; poll();
        };
        $("#set-apply").onclick = async () => {
          await api("/api/settings", { method: "POST", body: readCfg() });
          poll();
        };
        $("#set-save").onclick = () => saveDialog();
        $("#set-export").onclick = async () => {
          const data = await api("/api/export", { method: "POST", body: {} });
          const blob = new Blob([JSON.stringify(data)], { type: "application/json" });
          const a = document.createElement("a");
          a.href = URL.createObjectURL(blob);
          a.download = `${state.summary.name.replace(/\s+/g, "-")}.genesis.json`;
          a.click();
        };
        $("#set-import").onclick = () => $("#import-file").click();
        $("#import-file").onchange = async (e) => {
          const file = e.target.files[0];
          if (!file) return;
          const data = JSON.parse(await file.text());
          await api("/api/import", { method: "POST", body: data });
          lastVersion = -1; poll();
        };
      } else if (activeTab === "history") {
        const hist = await api("/api/history");
        el.innerHTML = `
          <h2>World History</h2>
          <canvas class="chart" id="chart-species"></canvas>
          <canvas class="chart" id="chart-animals"></canvas>
          ${hist.history.slice().reverse().map(h =>
            `<div class="event-item"><div>${h.text}</div><div class="when">${h.date} · ${h.kind}</div></div>`).join("") || "<p class='muted'>No history yet.</p>"}`;
        drawChart($("#chart-species"), hist.stats, "species", "#ffb84d");
        drawChart($("#chart-animals"), hist.stats, "total_animals", "#ff6b6b");
      }
    } catch (err) {
      el.innerHTML = `<p class="muted">Error loading panel: ${err.message}</p>`;
    }
  }

  function drawChart(canvas, stats, key, color) {
    if (!canvas || !stats || !stats.length) return;
    const ctx = canvas.getContext("2d");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = canvas.clientWidth * dpr; canvas.height = 90 * dpr;
    const vals = stats.map(s => s[key] || 0);
    const min = Math.min(...vals), max = Math.max(...vals);
    const range = max - min || 1;
    ctx.strokeStyle = color; ctx.lineWidth = 1.5 * dpr;
    ctx.beginPath();
    vals.forEach((v, i) => {
      const x = i / Math.max(vals.length - 1, 1) * canvas.width;
      const y = canvas.height - 6 * dpr - (v - min) / range * (canvas.height - 14 * dpr);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#8a9bb8"; ctx.font = `${10 * dpr}px sans-serif`;
    ctx.fillText(`${key}: ${min.toFixed(2)} … ${max.toFixed(2)}`, 6 * dpr, 12 * dpr);
  }

  /* ---------------- inspector ---------------- */
  async function inspect(lat, lon, silent) {
    try {
      const c = await api(`/api/cell?lat=${lat.toFixed(3)}&lon=${lon.toFixed(3)}`);
      selectedCell = { lat: c.lat, lon: c.lon };
      $("#inspector-empty").style.display = "none";
      const body = $("#inspector-body");
      body.style.display = "block";
      body.innerHTML = `
        <div class="kv"><span>Coordinates</span><span>${c.lat}°, ${c.lon}°</span></div>
        <div class="kv"><span>Elevation</span><span>${c.elevation_m} m</span></div>
        <div class="kv"><span>Biome</span><span>${c.biome}</span></div>
        <div class="kv"><span>Local season</span><span>${c.local_season || "—"}</span></div>
        <div class="kv"><span>Weather now</span><span>${c.weather}</span></div>
        <h3>Climate</h3>
        <div class="kv"><span>Temperature</span><span>${c.temp_c}°C</span></div>
        <div class="kv"><span>Precipitation</span><span>${c.precip_mm_year} mm/yr</span></div>
        <div class="kv"><span>Rain today</span><span>${c.precip_today_mm} mm</span></div>
        <div class="kv"><span>Humidity</span><span>${c.humidity_pct}%</span></div>
        <div class="kv"><span>Pressure</span><span>${c.pressure_hpa} hPa</span></div>
        <div class="kv"><span>Wind</span><span>${c.wind_ms} m/s @ ${c.wind_dir_deg}°</span></div>
        <div class="kv"><span>Cloud</span><span>${c.cloud_pct}%</span></div>
        <h3>Soil</h3>
        <div class="kv"><span>Fertility</span><span>${c.soil.fertility}</span></div>
        <div class="kv"><span>Moisture</span><span>${c.soil.moisture}</span></div>
        <div class="kv"><span>Nutrients</span><span>${c.soil.nutrients}</span></div>
        <div class="kv"><span>Organic matter</span><span>${c.soil.organic}</span></div>
        <div class="kv"><span>Acidity</span><span>pH ${c.soil.acidity_ph}</span></div>
        <div class="kv"><span>Depth</span><span>${c.soil.depth_cm} cm</span></div>
        <div class="kv"><span>Erosion</span><span>${c.soil.erosion}</span></div>
        <h3>Vegetation</h3>
        <div class="kv"><span>Cover</span><span>${(c.vegetation.cover * 100).toFixed(0)}%</span></div>
        <div class="kv"><span>Biomass</span><span>${c.vegetation.biomass_tha} t/ha</span></div>
        ${Object.entries(c.vegetation.groups).map(([g, v]) => `<div class="kv"><span>${g}</span><span>${(v * 100).toFixed(0)}%</span></div>`).join("")}
        <h3>Water</h3>
        <div class="kv"><span>River</span><span>${c.water.river ? "order " + c.water.river_order : "—"}</span></div>
        <div class="kv"><span>Flow accumulation</span><span>${c.water.flow_accum}</span></div>
        <div class="kv"><span>Lake / Wetland / Glacier</span><span>${c.water.lake ? "lake" : c.water.wetland ? "wetland" : c.water.glacier ? "glacier" : "—"}</span></div>
        <div class="kv"><span>Distance to ocean</span><span>${c.water.dist_ocean_km} km</span></div>
        <h3>Resources</h3>
        ${Object.entries(c.resources).map(([r, v]) => `<div class="kv"><span>${r}</span><span>${v}</span></div>`).join("")}
        <h3>Fauna here</h3>
        ${c.fauna.length ? c.fauna.map(f => `<span class="chip">${f.species} · ${f.population}</span>`).join("") : "<p class='muted'>No animal populations in this region.</p>"}
        ${c.drought_days > 0 ? `<div class="kv"><span>Drought days</span><span>${c.drought_days}</span></div>` : ""}
        ${c.fire_days > 0 ? `<div class="kv"><span>Fire days</span><span>${c.fire_days}</span></div>` : ""}`;
    } catch (e) { if (!silent) console.warn(e); }
  }

  /* ---------------- save / load dialogs ---------------- */
  function openModal(title, html) {
    $("#modal-title").textContent = title;
    $("#modal-body").innerHTML = html;
    $("#modal").style.display = "flex";
  }
  $("#modal-close").onclick = () => $("#modal").style.display = "none";
  $("#modal").onclick = (e) => { if (e.target.id === "modal") $("#modal").style.display = "none"; };

  async function saveDialog() {
    openModal("Save World", `
      <label class="settings-label">Save name</label>
      <input type="text" id="save-name" value="${state.summary.name}">
      <div class="btn-row"><button id="save-go" class="btn-primary">Save</button></div>
      <p class="muted">Stored locally in the server's saves/ folder. No cloud involved.</p>`);
    $("#save-go").onclick = async () => {
      const name = $("#save-name").value;
      await api("/api/save", { method: "POST", body: { name } });
      $("#modal").style.display = "none";
    };
  }
  $("#btn-save").onclick = saveDialog;

  $("#btn-load").onclick = async () => {
    const { saves } = await api("/api/saves");
    openModal("Load World", saves.length ? saves.map(s => `
      <div class="save-item">
        <div><b>${s.name}</b><br><span class="muted">seed ${s.seed} · day ${Math.round(s.sim_day)}</span></div>
        <div>
          <button data-load="${s.name}" class="btn-primary">Load</button>
          <button data-del="${s.name}">Delete</button>
        </div>
      </div>`).join("") : "<p class='muted'>No saves yet. Use Save first.</p>");
    $$("#modal-body [data-load]").forEach(b => b.onclick = async () => {
      await api("/api/load", { method: "POST", body: { name: b.dataset.load } });
      $("#modal").style.display = "none";
      lastVersion = -1; poll();
    });
    $$("#modal-body [data-del]").forEach(b => b.onclick = async () => {
      await api("/api/delete-save", { method: "POST", body: { name: b.dataset.del } });
      $("#btn-load").click();
    });
  };

  /* ---------------- boot ---------------- */
  poll();
})();
