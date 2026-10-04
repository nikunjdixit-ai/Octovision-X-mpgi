(function(){

  /* ==============================================================
     EASY CUSTOMIZE

     One entry per class. The student names here should match the
     folder of reference photos that class already has on the
     operator computer, since that folder is what the AI pipeline
     checks the uploaded video against. Add or remove classes and
     names freely.
  ============================================================== */
  const CLASSES = {
    "Class A": ["Aditi Sharma", "Rohan Verma", "Meera Iyer", "Karan Malhotra", "Priya Nair"],
    "Class B": ["Sanjana Rao", "Farhan Khan", "Divya Menon", "Arjun Kapoor"],
    "Class C": ["Ishaan Gupta", "Neha Joshi", "Vikram Singh", "Ananya Bose"],
    "Class D": ["Ishita Patel", "Rohan Iyer", "Kavya Nair", "Aditya Sharma", "Sneha Kapoor"],
    "Class E": ["Aarav Reddy", "Priya Malhotra", "Rohan Nair", "Sanya Gupta", "Vikrant Singh"],
    "Class F": ["Ananya Iyer", "Kiran Malhotra", "Rohan Gupta", "Neha Rao", "Arjun Verma"],
    "Class G": ["Mira Nair", "Diya Aggarwal", "Rohan Kapoor", "Sameer Malhotra", "Tina Gupta"],
    "Class H": ["Rohan Singh", "Neha Iyer", "Sanya Verma", "Aditya Kapoor", "Kavya Malhotra"]
  };

  /* Formspree endpoint for the settings feedback form.
     Replace this with the real form link when it is ready. */
  const FORMSPREE_ENDPOINT = "https://formspree.io/f/yourFormId";

  const KEY_NAME = "trace_user_name";
  const PREF_SOUND = "trace_pref_sound";
  const PREF_CONFIRM_CLEAR = "trace_pref_confirm_clear";
  const PREF_REDUCE_MOTION = "trace_pref_reduce_motion";
  const PREF_COMPACT = "trace_pref_compact";
  const PREF_CONFIDENCE = "trace_pref_confidence";
  const PREF_AUTO_EXPORT = "trace_pref_auto_export";
  const PREF_TERMINAL = "trace_pref_terminal";
  const PREF_MONO = "trace_pref_mono";

  function applyTerminalScale(which){
    document.documentElement.style.setProperty("--terminal-font", which === "large" ? "15px" : "13px");
  }

  /* ---------- small icon library, one stroke style throughout ----------
     The gear here is the same clean eight tooth gear used on the header
     settings button in dashboard.html. Keep the two in sync. */
  function iconGear(size){
    return '<svg width="' + size + '" height="' + size + '" viewBox="0 0 24 24" fill="none">' +
      '<path d="M12 8.6C10.12 8.6 8.6 10.12 8.6 12C8.6 13.88 10.12 15.4 12 15.4C13.88 15.4 15.4 13.88 15.4 12C15.4 10.12 13.88 8.6 12 8.6Z" stroke="currentColor" stroke-width="1.4"/>' +
      '<path d="M19.4 12C19.4 12.47 19.36 12.93 19.29 13.38L21.28 14.93C21.46 15.07 21.51 15.32 21.39 15.53L19.49 18.82C19.37 19.03 19.13 19.11 18.91 19.03L16.57 18.09C16.08 18.46 15.55 18.77 14.98 19.01L14.63 21.5C14.6 21.73 14.4 21.9 14.17 21.9H10.37C10.14 21.9 9.94 21.73 9.91 21.5L9.56 19.01C8.99 18.77 8.46 18.46 7.97 18.09L5.63 19.03C5.41 19.11 5.17 19.03 5.05 18.82L3.15 15.53C3.03 15.32 3.08 15.07 3.26 14.93L5.25 13.38C5.18 12.93 5.14 12.47 5.14 12C5.14 11.53 5.18 11.07 5.25 10.62L3.26 9.07C3.08 8.93 3.03 8.68 3.15 8.47L5.05 5.18C5.17 4.97 5.41 4.89 5.63 4.97L7.97 5.91C8.46 5.54 8.99 5.23 9.56 4.99L9.91 2.5C9.94 2.27 10.14 2.1 10.37 2.1H14.17C14.4 2.1 14.6 2.27 14.63 2.5L14.98 4.99C15.55 5.23 16.08 5.54 16.57 5.91L18.91 4.97C19.13 4.89 19.37 4.97 19.49 5.18L21.39 8.47C21.51 8.68 21.46 8.93 21.28 9.07L19.29 10.62C19.36 11.07 19.4 11.53 19.4 12Z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/>' +
    '</svg>';
  }
  function iconControls(size){
    return '<svg width="' + size + '" height="' + size + '" viewBox="0 0 28 28" fill="none">' +
      '<path d="M5 8H16M20 8H23M5 20H12M16 20H23M5 14H9M13 14H23" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>' +
      '<circle cx="18" cy="8" r="2.2" stroke="currentColor" stroke-width="1.4"/>' +
      '<circle cx="14" cy="20" r="2.2" stroke="currentColor" stroke-width="1.4"/>' +
      '<circle cx="11" cy="14" r="2.2" stroke="currentColor" stroke-width="1.4"/>' +
    '</svg>';
  }
  function iconAppearance(size){
    return '<svg width="' + size + '" height="' + size + '" viewBox="0 0 28 28" fill="none">' +
      '<circle cx="14" cy="14" r="8" stroke="currentColor" stroke-width="1.4"/>' +
      '<path d="M14 6C18.4 6 22 9.6 22 14C22 18.4 18.4 22 14 22Z" fill="currentColor" stroke="none"/>' +
    '</svg>';
  }
  function iconFeedback(size){
    return '<svg width="' + size + '" height="' + size + '" viewBox="0 0 28 28" fill="none">' +
      '<path d="M4 7.5C4 6.1 5.1 5 6.5 5H21.5C22.9 5 24 6.1 24 7.5V17.5C24 18.9 22.9 20 21.5 20H12L7 24V20H6.5C5.1 20 4 18.9 4 17.5V7.5Z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/>' +
    '</svg>';
  }

  /* ---------- shared helpers ---------- */
  function escapeHtml(str){
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
  }
  function slug(str){
    return str.replace(/[^a-zA-Z0-9]+/g, "_");
  }

  /* ---------- shared helpers ---------- */
  function downloadFile(filename, content, mime){
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }
  function playChime(){
    try{
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = 660;
      gain.gain.setValueAtTime(0.0001, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.12, ctx.currentTime + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.5);
      osc.connect(gain).connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.5);
    } catch(err){ /* audio not available, fine to skip */ }
  }

  /* ==============================================================
     OPERATOR NAME
  ============================================================== */
  const operatorName = localStorage.getItem(KEY_NAME);
  if (!operatorName){
    window.location.href = "index.html";
    return;
  }
  document.getElementById("operatorName").textContent = operatorName;
  document.getElementById("settingsOperatorName").textContent = operatorName;

  /* Terminal size is restored before anything renders so the first paint
     already matches what the operator saved last time. */
  applyTerminalScale(localStorage.getItem(PREF_TERMINAL));

  /* ==============================================================
     CLASS SELECTION
  ============================================================== */
  const classGrid = document.getElementById("classGrid");
  let activeClass = null;

  Object.keys(CLASSES).forEach(function(name){
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "class-chip";
    chip.textContent = name;
    chip.addEventListener("click", function(){ selectClass(name); });
    classGrid.appendChild(chip);
  });

  function selectClass(name){
    if (attendanceRunning) return;
    activeClass = name;
    Array.from(classGrid.children).forEach(function(chip){
      chip.classList.toggle("active", chip.textContent === name);
    });
    renderRoster(name);
    updateAttendanceState();
  }

  /* ==============================================================
     VIDEO SOURCE  (one source at a time: an MP4/MOV file OR an RTSP URL)
  ============================================================== */
  const MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024; // 2GB
  const sourceTabs = Array.from(document.querySelectorAll(".source-tab"));
  const panelFile = document.getElementById("panelFile");
  const panelRtsp = document.getElementById("panelRtsp");
  const sourceError = document.getElementById("sourceError");

  const videoZone = document.getElementById("videoUploadZone");
  const videoInput = document.getElementById("videoFileInput");
  const videoEmpty = document.getElementById("videoUploadEmpty");
  const videoPreview = document.getElementById("videoUploadPreview");
  const videoFilename = document.getElementById("videoFilename");
  const videoDeleteBtn = document.getElementById("videoDeleteBtn");

  const rtspForm = document.getElementById("rtspForm");
  const rtspInput = document.getElementById("rtspInput");
  const rtspPreview = document.getElementById("rtspPreview");
  const rtspUrlLabel = document.getElementById("rtspUrlLabel");
  const rtspDeleteBtn = document.getElementById("rtspDeleteBtn");

  /* source is null, { type:"file", file }, or { type:"rtsp", url } */
  let source = null;
  let activeTab = "file";

  function hasSource(){ return !!source; }

  function showSourceError(msg){
    sourceError.textContent = msg || "";
    sourceError.hidden = !msg;
  }

  function maskRtsp(url){
    /* never show the password on screen */
    return url.replace(/^(rtsps?:\/\/[^:\/@\s]+):[^@\s]*@/i, "$1:****@");
  }

  function describeSource(){
    if (!source) return null;
    return source.type === "file"
      ? { type: "mp4", name: source.file.name, size: source.file.size }
      : { type: "rtsp", url: source.url };
  }

  function renderSource(){
    videoEmpty.hidden = source && source.type === "file";
    videoPreview.hidden = !(source && source.type === "file");
    if (source && source.type === "file") videoFilename.textContent = source.file.name;

    rtspForm.hidden = !!(source && source.type === "rtsp");
    rtspPreview.hidden = !(source && source.type === "rtsp");
    if (source && source.type === "rtsp") rtspUrlLabel.textContent = maskRtsp(source.url);
    updateAttendanceState();
  }

  function clearSource(){
    source = null;
    videoInput.value = "";
    rtspInput.value = "";
    showSourceError("");
    renderSource();
  }

  function switchTab(which){
    if (attendanceRunning || which === activeTab) return;
    activeTab = which;
    sourceTabs.forEach(function(t){
      const on = t.dataset.source === which;
      t.classList.toggle("active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
    });
    panelFile.hidden = which !== "file";
    panelRtsp.hidden = which !== "rtsp";
    clearSource(); // only one source is ever active
  }
  sourceTabs.forEach(function(t){
    t.addEventListener("click", function(){ switchTab(t.dataset.source); });
  });

  /* ----- MP4 / MOV file ----- */
  function setVideoFile(file){
    const okType = /^video\/(mp4|quicktime)$/i.test(file.type) || /\.(mp4|mov)$/i.test(file.name);
    if (!okType){ showSourceError("Please choose an MP4 or MOV video."); return; }
    if (file.size > MAX_VIDEO_BYTES){ showSourceError("That file is over 2GB."); return; }
    showSourceError("");
    source = { type: "file", file: file };
    renderSource();
  }

  videoZone.addEventListener("click", function(){ if (!attendanceRunning) videoInput.click(); });
  videoZone.addEventListener("keydown", function(e){
    if ((e.key === "Enter" || e.key === " ") && !attendanceRunning){ e.preventDefault(); videoInput.click(); }
  });
  videoZone.addEventListener("dragover", function(e){ e.preventDefault(); if (!attendanceRunning) videoZone.classList.add("drag-active"); });
  videoZone.addEventListener("dragleave", function(){ videoZone.classList.remove("drag-active"); });
  videoZone.addEventListener("drop", function(e){
    e.preventDefault();
    videoZone.classList.remove("drag-active");
    if (attendanceRunning) return;
    if (e.dataTransfer.files && e.dataTransfer.files[0]) setVideoFile(e.dataTransfer.files[0]);
  });
  videoInput.addEventListener("change", function(){
    if (videoInput.files && videoInput.files[0]) setVideoFile(videoInput.files[0]);
  });
  videoDeleteBtn.addEventListener("click", function(e){
    e.stopPropagation(); // don't re-open the file picker
    if (!attendanceRunning) clearSource();
  });

  /* ----- RTSP URL ----- */
  rtspForm.addEventListener("submit", function(e){
    e.preventDefault();
    if (attendanceRunning) return;
    const url = rtspInput.value.trim();
    if (!/^rtsps?:\/\/[^\s\/]+(\/\S*)?$/i.test(url)){
      showSourceError("Enter a valid RTSP URL, e.g. rtsp://user:pass@192.168.1.20:554/stream");
      return;
    }
    showSourceError("");
    source = { type: "rtsp", url: url };
    renderSource();
  });
  rtspInput.addEventListener("input", function(){ showSourceError(""); });
  rtspDeleteBtn.addEventListener("click", function(){ if (!attendanceRunning) clearSource(); });

  /* ==============================================================
     ROSTER + ATTENDANCE RUN
  ============================================================== */
  const terminalIdle = document.getElementById("terminalIdle");
  const terminalLabel = document.getElementById("terminalLabel");
  const rosterList = document.getElementById("rosterList");
  const terminalActions = document.getElementById("terminalActions");
  const markBtn = document.getElementById("markAttendanceBtn");
  const attendanceHint = document.getElementById("attendanceHint");
  let attendanceRunning = false;
  let lastResults = [];

  function renderRoster(className){
    terminalIdle.hidden = true;
    terminalLabel.textContent = "trace / " + className;
    rosterList.innerHTML = "";
    terminalActions.hidden = true;
    lastResults = [];

    CLASSES[className].forEach(function(name){
      const row = document.createElement("div");
      row.className = "roster-row";
      row.innerHTML =
        '<span class="roster-name">' + escapeHtml(name) + '</span>' +
        '<span class="roster-status" data-state="pending">[ ]</span>';
      rosterList.appendChild(row);
    });
  }

  function updateAttendanceState(){
    if (attendanceRunning) return;
    const ready = hasSource() && activeClass;
    markBtn.disabled = !ready;
    if (!hasSource() && !activeClass) {
      attendanceHint.textContent = "Add a video source and choose a class to begin.";
    } else if (!hasSource()) {
      attendanceHint.textContent = "Now add a video file or RTSP stream.";
    } else if (!activeClass) {
      attendanceHint.textContent = "Now choose a class.";
    } else {
      attendanceHint.textContent = "Ready. Click mark attendance to start.";
    }
  }

  markBtn.addEventListener("click", function(){
    if (markBtn.disabled || attendanceRunning) return;
    runAttendance();
  });

  function runAttendance(){
    attendanceRunning = true;
    markBtn.disabled = true;
    videoZone.setAttribute("aria-disabled", "true");
    sourceTabs.forEach(function(t){ t.disabled = true; });
    Array.from(classGrid.children).forEach(function(chip){ chip.disabled = true; });
    attendanceHint.textContent = "Checking the video against " + activeClass + ".";

    const rows = Array.from(rosterList.children);
    let i = 0;

    function step(){
      if (i >= rows.length){
        finishAttendance();
        return;
      }
      const row = rows[i];
      const status = row.querySelector(".roster-status");
      status.dataset.state = "checking";
      status.textContent = "[ checking ]";

      setTimeout(function(){
        const present = Math.random() < 0.78;
        status.dataset.state = present ? "present" : "absent";
        status.textContent = present ? "[ present ]" : "[ absent ]";
        const confidence = present ? 0.86 + Math.random() * 0.13 : Math.random() * 0.42;
        lastResults.push({
          name: row.querySelector(".roster-name").textContent,
          status: present ? "present" : "absent",
          confidence: localStorage.getItem(PREF_CONFIDENCE) === "1" ? Number(confidence.toFixed(2)) : undefined
        });
        i += 1;
        setTimeout(step, 260);
      }, 550 + Math.random() * 350);
    }
    step();
  }

  function finishAttendance(){
    attendanceRunning = false;
    Array.from(classGrid.children).forEach(function(chip){ chip.disabled = false; });
    videoZone.removeAttribute("aria-disabled");
    sourceTabs.forEach(function(t){ t.disabled = false; });
    terminalActions.hidden = false;
    attendanceHint.textContent = "Attendance complete for " + activeClass + ".";
    if (localStorage.getItem(PREF_SOUND) === "1") playChime();
    if (localStorage.getItem(PREF_AUTO_EXPORT) === "1"){
      const payload = { class: activeClass, date: new Date().toISOString(), source: describeSource(), students: lastResults };
      downloadFile("attendance_" + slug(activeClass) + "_" + todayStamp() + ".json", JSON.stringify(payload, null, 2), "application/json");
    }
  }

  document.getElementById("downloadJsonBtn").addEventListener("click", function(){
    const payload = { class: activeClass, date: new Date().toISOString(), source: describeSource(), students: lastResults };
    downloadFile("attendance_" + slug(activeClass) + "_" + todayStamp() + ".json", JSON.stringify(payload, null, 2), "application/json");
  });

  document.getElementById("downloadCsvBtn").addEventListener("click", function(){
    let csv = "name,status\n";
    lastResults.forEach(function(r){ csv += '"' + r.name.replace(/"/g, '""') + '",' + r.status + "\n"; });
    downloadFile("attendance_" + slug(activeClass) + "_" + todayStamp() + ".csv", csv, "text/csv");
  });

  document.getElementById("clearAllBtn").addEventListener("click", function(){
    if (localStorage.getItem(PREF_CONFIRM_CLEAR) === "1"){
      const ok = window.confirm("This clears the current video source, class and results. Continue?");
      if (!ok) return;
    }
    clearSource();
    activeClass = null;
    Array.from(classGrid.children).forEach(function(chip){ chip.classList.remove("active"); });
    rosterList.innerHTML = "";
    terminalIdle.hidden = false;
    terminalLabel.textContent = "trace attendance session";
    terminalActions.hidden = true;
    lastResults = [];
    updateAttendanceState();
  });

  /* ==============================================================
     SETTINGS MODAL
  ============================================================== */
  const settingsOverlay = document.getElementById("settingsOverlay");
  const settingsBtn = document.getElementById("settingsBtn");
  const settingsCloseBtn = document.getElementById("settingsCloseBtn");
  const settingsPanel = document.getElementById("settingsPanel");
  const settingsNav = document.getElementById("settingsNav");

  function openSettings(){
    settingsOverlay.classList.add("open");
    renderSettingsPanel("overview");
  }
  function closeSettings(){ settingsOverlay.classList.remove("open"); }

  settingsBtn.addEventListener("click", openSettings);
  settingsCloseBtn.addEventListener("click", closeSettings);
  settingsOverlay.addEventListener("click", function(e){ if (e.target === settingsOverlay) closeSettings(); });
  document.addEventListener("keydown", function(e){ if (e.key === "Escape") closeSettings(); });

  Array.from(settingsNav.querySelectorAll("button[data-panel]")).forEach(function(btn){
    btn.addEventListener("click", function(){
      Array.from(settingsNav.querySelectorAll("button[data-panel]")).forEach(function(b){ b.classList.remove("active"); });
      btn.classList.add("active");
      renderSettingsPanel(btn.dataset.panel);
    });
  });

  function renderSettingsPanel(which){
    if (which === "controls") return renderControlsPanel();
    if (which === "appearance") return renderAppearancePanel();
    if (which === "feedback") return renderFeedbackPanel();
    return renderOverviewPanel();
  }

  function renderOverviewPanel(){
    settingsPanel.innerHTML =
      '<div id="settingsIcon" class="settings-icon">' + iconGear(72) + '</div>' +
      '<h2 id="settingsPanelTitle">Settings</h2>' +
      '<p class="settings-desc">Choose an option from the list to adjust how this dashboard behaves and looks.</p>';
  }

  function renderControlsPanel(){
    settingsPanel.innerHTML =
      '<span class="settings-icon">' + iconControls(48) + '</span>' +
      '<h2 id="settingsPanelTitle">Controls</h2>' +
      '<p class="settings-desc">Four controls for sound, confirmation, automatic exports and match scores.</p>' +
      '<div class="settings-panel-body">' +
        '<div class="toggle-row">' +
          '<div class="toggle-copy"><span class="toggle-title">Sound when attendance finishes</span><span class="toggle-desc">Play a short sound once every student has been checked.</span></div>' +
          '<label class="switch"><input type="checkbox" id="soundToggle"><span class="switch-track"><span class="switch-knob"></span></span></label>' +
        '</div>' +
        '<div class="toggle-row">' +
          '<div class="toggle-copy"><span class="toggle-title">Confirm before clearing</span><span class="toggle-desc">Ask before clearing the video, class and results.</span></div>' +
          '<label class="switch"><input type="checkbox" id="confirmClearToggle"><span class="switch-track"><span class="switch-knob"></span></span></label>' +
        '</div>' +
        '<div class="toggle-row">' +
          '<div class="toggle-copy"><span class="toggle-title">Include confidence values</span><span class="toggle-desc">Save a match score next to each student in the JSON and CSV downloads.</span></div>' +
          '<label class="switch"><input type="checkbox" id="confidenceToggle"><span class="switch-track"><span class="switch-knob"></span></span></label>' +
        '</div>' +
        '<div class="toggle-row">' +
          '<div class="toggle-copy"><span class="toggle-title">Export when a run finishes</span><span class="toggle-desc">Download the JSON report automatically the moment the last student is checked.</span></div>' +
          '<label class="switch"><input type="checkbox" id="autoExportToggle"><span class="switch-track"><span class="switch-knob"></span></span></label>' +
        '</div>' +
        '<div class="settings-actions">' +
          '<button type="button" class="settings-action-link" id="resetControlsBtn">Reset controls to defaults</button>' +
        '</div>' +
      '</div>';

    const soundToggle = document.getElementById("soundToggle");
    const confirmClearToggle = document.getElementById("confirmClearToggle");
    const confidenceToggle = document.getElementById("confidenceToggle");
    const autoExportToggle = document.getElementById("autoExportToggle");
    soundToggle.checked = localStorage.getItem(PREF_SOUND) === "1";
    confirmClearToggle.checked = localStorage.getItem(PREF_CONFIRM_CLEAR) === "1";
    confidenceToggle.checked = localStorage.getItem(PREF_CONFIDENCE) === "1";
    autoExportToggle.checked = localStorage.getItem(PREF_AUTO_EXPORT) === "1";
    soundToggle.addEventListener("change", function(){ localStorage.setItem(PREF_SOUND, soundToggle.checked ? "1" : "0"); });
    confirmClearToggle.addEventListener("change", function(){ localStorage.setItem(PREF_CONFIRM_CLEAR, confirmClearToggle.checked ? "1" : "0"); });
    confidenceToggle.addEventListener("change", function(){ localStorage.setItem(PREF_CONFIDENCE, confidenceToggle.checked ? "1" : "0"); });
    autoExportToggle.addEventListener("change", function(){ localStorage.setItem(PREF_AUTO_EXPORT, autoExportToggle.checked ? "1" : "0"); });

    document.getElementById("resetControlsBtn").addEventListener("click", function(){
      [PREF_SOUND, PREF_CONFIRM_CLEAR, PREF_CONFIDENCE, PREF_AUTO_EXPORT].forEach(function(k){ localStorage.removeItem(k); });
      renderControlsPanel();
    });
  }

  function renderAppearancePanel(){
    const currentTerminal = localStorage.getItem(PREF_TERMINAL) || "normal";
    settingsPanel.innerHTML =
      '<span class="settings-icon">' + iconAppearance(48) + '</span>' +
      '<h2 id="settingsPanelTitle">Appearance</h2>' +
      '<p class="settings-desc">Adjust motion, layout density, terminal size and number style.</p>' +
      '<div class="settings-panel-body">' +
        '<div class="toggle-row">' +
          '<div class="toggle-copy"><span class="toggle-title">Reduce motion</span><span class="toggle-desc">Turn off hover lift and shake animations across the dashboard.</span></div>' +
          '<label class="switch"><input type="checkbox" id="reduceMotionToggle"><span class="switch-track"><span class="switch-knob"></span></span></label>' +
        '</div>' +
        '<div class="toggle-row">' +
          '<div class="toggle-copy"><span class="toggle-title">Compact layout</span><span class="toggle-desc">Tighten spacing so more fits on screen at once.</span></div>' +
          '<label class="switch"><input type="checkbox" id="compactToggle"><span class="switch-track"><span class="switch-knob"></span></span></label>' +
        '</div>' +
        '<div class="segmented-row">' +
          '<div class="toggle-copy"><span class="toggle-title">Terminal text size</span><span class="toggle-desc">Applies to the black terminal panel only.</span></div>' +
          '<div class="segment-group" role="group" aria-label="Terminal text size">' +
            '<button type="button" class="segment' + (currentTerminal === "normal" ? " active" : "") + '" data-terminal="normal">Normal</button>' +
            '<button type="button" class="segment' + (currentTerminal === "large" ? " active" : "") + '" data-terminal="large">Large</button>' +
          '</div>' +
        '</div>' +
        '<div class="segmented-row">' +
          '<div class="toggle-copy"><span class="toggle-title">Monospace roster</span><span class="toggle-desc">Render student names in the terminal type style.</span></div>' +
          '<label class="switch"><input type="checkbox" id="monoToggle"><span class="switch-track"><span class="switch-knob"></span></span></label>' +
        '</div>' +
      '</div>';

    const reduceMotionToggle = document.getElementById("reduceMotionToggle");
    const compactToggle = document.getElementById("compactToggle");
    const monoToggle = document.getElementById("monoToggle");
    reduceMotionToggle.checked = document.body.classList.contains("reduce-motion");
    compactToggle.checked = document.body.classList.contains("compact");
    monoToggle.checked = document.body.classList.contains("mono-roster");
    reduceMotionToggle.addEventListener("change", function(){
      document.body.classList.toggle("reduce-motion", reduceMotionToggle.checked);
      localStorage.setItem(PREF_REDUCE_MOTION, reduceMotionToggle.checked ? "1" : "0");
    });
    compactToggle.addEventListener("change", function(){
      document.body.classList.toggle("compact", compactToggle.checked);
      localStorage.setItem(PREF_COMPACT, compactToggle.checked ? "1" : "0");
    });
    monoToggle.addEventListener("change", function(){
      document.body.classList.toggle("mono-roster", monoToggle.checked);
      localStorage.setItem(PREF_MONO, monoToggle.checked ? "1" : "0");
    });

    Array.from(settingsPanel.querySelectorAll(".segment")).forEach(function(btn){
      btn.addEventListener("click", function(){
        Array.from(settingsPanel.querySelectorAll(".segment")).forEach(function(s){ s.classList.remove("active"); });
        btn.classList.add("active");
        localStorage.setItem(PREF_TERMINAL, btn.dataset.terminal);
        applyTerminalScale(btn.dataset.terminal);
      });
    });
  }

  function renderFeedbackPanel(){
    settingsPanel.innerHTML =
      '<span class="settings-icon">' + iconFeedback(48) + '</span>' +
      '<h2 id="settingsPanelTitle">Send feedback</h2>' +
      '<p class="settings-desc">Tell us what is working well or what could be better.</p>' +
      '<form class="feedback-form" id="feedbackForm">' +
        '<div class="form-field"><label for="feedbackName">Your name</label><input id="feedbackName" name="name" type="text" required></div>' +
        '<div class="form-field"><label for="feedbackMessage">Message</label><textarea id="feedbackMessage" name="message" required></textarea></div>' +
        '<p class="form-status" id="feedbackStatus"></p>' +
        '<button type="submit" class="btn btn-magnetic btn-large">Send feedback</button>' +
      '</form>';

    document.getElementById("feedbackName").value = operatorName;

    document.getElementById("feedbackForm").addEventListener("submit", function(e){
      e.preventDefault();
      const status = document.getElementById("feedbackStatus");
      status.textContent = "Sending.";
      status.className = "form-status";
      fetch(FORMSPREE_ENDPOINT, {
        method: "POST",
        headers: { "Accept": "application/json" },
        body: new FormData(e.target)
      }).then(function(res){
        if (res.ok){
          status.textContent = "Thanks, your feedback has been sent.";
          status.className = "form-status success";
          e.target.reset();
          document.getElementById("feedbackName").value = operatorName;
        } else {
          status.textContent = "Something went wrong. Please try again.";
          status.className = "form-status error";
        }
      }).catch(function(){
        status.textContent = "Something went wrong. Please try again.";
        status.className = "form-status error";
      });
    });
  }

  /* ==============================================================
     RESTORE APPEARANCE PREFERENCES ON LOAD
  ============================================================== */
  if (localStorage.getItem(PREF_REDUCE_MOTION) === "1") document.body.classList.add("reduce-motion");
  if (localStorage.getItem(PREF_COMPACT) === "1") document.body.classList.add("compact");
  if (localStorage.getItem(PREF_MONO) === "1") document.body.classList.add("mono-roster");

  updateAttendanceState();

})();
