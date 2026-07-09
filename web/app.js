const state = {
  view: 'chat',
  config: null,
  sessions: [],
  jobs: [],
  graph: null,
  backpack: null,
  selectedSessionId: localStorage.getItem('semantic_ants.session') || 'default',
  selectedNodeId: null,
  selectedEdgeId: null,
  nodeDetail: null,
  activeDetailKind: 'none',
  lastResult: null,
  graphQuery: '',
  graphLimit: 160,
  busy: false,
  trainingText: localStorage.getItem('semantic_ants.training_text') || '',
  trainSessionId: localStorage.getItem('semantic_ants.train_session') || 'default',
  trainEpochs: 1,
  graphInstances: new Map(),
  visualRequestId: 0,
};

const el = {
  statusStrip: document.getElementById('statusStrip'),
  tabs: Array.from(document.querySelectorAll('.tab')),
  views: Array.from(document.querySelectorAll('[data-view-panel]')),
  chatSessionInput: document.getElementById('chatSessionInput'),
  resetSessionButton: document.getElementById('resetSessionButton'),
  chatFeed: document.getElementById('chatFeed'),
  chatInput: document.getElementById('chatInput'),
  chatForm: document.getElementById('chatForm'),
  chatHint: document.getElementById('chatHint'),
  backpackStats: document.getElementById('backpackStats'),
  backpackGraph: document.getElementById('backpackGraph'),
  chatDetail: document.getElementById('chatDetail'),
  graphQueryInput: document.getElementById('graphQueryInput'),
  graphLimitInput: document.getElementById('graphLimitInput'),
  graphRefreshButton: document.getElementById('graphRefreshButton'),
  globalGraph: document.getElementById('globalGraph'),
  graphDetail: document.getElementById('graphDetail'),
  trainSessionInput: document.getElementById('trainSessionInput'),
  trainEpochsInput: document.getElementById('trainEpochsInput'),
  trainInput: document.getElementById('trainInput'),
  trainButton: document.getElementById('trainButton'),
  jobsCount: document.getElementById('jobsCount'),
  jobsList: document.getElementById('jobsList'),
  backpackUpButton: document.getElementById('backpackUpButton'),
  backpackDownButton: document.getElementById('backpackDownButton'),
};

let jobTimer = null;

if (typeof window !== 'undefined') {
  window.__semanticAntsDebug = {
    getGraphInstance(id) {
      return state.graphInstances.get(String(id || ''))?.cy || null;
    },
    getState() {
      return state;
    },
  };
}

boot().catch((error) => {
  console.error(error);
  renderError(error);
});

async function boot() {
  wireEvents();
  el.chatSessionInput.value = state.selectedSessionId;
  el.trainSessionInput.value = state.trainSessionId;
  el.trainInput.value = state.trainingText;
  el.chatHint.textContent = 'Paste training text first if the graph is empty.';
  await Promise.all([
    refreshConfig(),
    refreshSessions(),
    refreshGraph(),
    refreshJobs(),
  ]);
  renderAll();
  startJobsPolling();
}

function wireEvents() {
  el.tabs.forEach((button) => {
    button.addEventListener('click', () => setView(button.dataset.view || 'chat'));
  });

  el.chatSessionInput.addEventListener('change', () => {
    const nextSessionId = sanitizeSessionId(el.chatSessionInput.value || 'default');
    if (nextSessionId !== state.selectedSessionId) {
      clearChatWorkspace();
    }
    state.selectedSessionId = nextSessionId;
    localStorage.setItem('semantic_ants.session', state.selectedSessionId);
    renderChat();
  });

  el.resetSessionButton.addEventListener('click', async () => {
    const sessionId = sanitizeSessionId(el.chatSessionInput.value || state.selectedSessionId);
    await requestJSON(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
    await refreshSessions();
    clearChatWorkspace();
    renderAll();
  });

  el.chatForm.addEventListener('submit', sendChatMessage);
  el.chatInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      sendChatMessage(event);
    }
  });

  el.graphRefreshButton.addEventListener('click', refreshGraph);
  el.graphQueryInput.addEventListener('change', () => {
    state.graphQuery = el.graphQueryInput.value.trim();
  });
  el.graphLimitInput.addEventListener('change', () => {
    state.graphLimit = clampInt(el.graphLimitInput.value, 24, 1000, 160);
  });

  el.trainSessionInput.addEventListener('change', () => {
    state.trainSessionId = sanitizeSessionId(el.trainSessionInput.value || 'default');
    localStorage.setItem('semantic_ants.train_session', state.trainSessionId);
  });
  el.trainEpochsInput.addEventListener('change', () => {
    state.trainEpochs = clampInt(el.trainEpochsInput.value, 1, 100, 1);
  });
  el.trainInput.addEventListener('input', () => {
    state.trainingText = el.trainInput.value;
    localStorage.setItem('semantic_ants.training_text', state.trainingText);
  });
  el.trainButton.addEventListener('click', startTraining);

  el.backpackUpButton.addEventListener('click', drillBackpackUp);
  el.backpackDownButton.addEventListener('click', drillBackpackDown);
}

function setView(view) {
  state.view = view;
  el.tabs.forEach((button) => button.classList.toggle('active', (button.dataset.view || 'chat') === view));
  el.views.forEach((panel) => panel.classList.toggle('view-active', panel.dataset.viewPanel === view));
  renderAll();
}

async function refreshConfig() {
  state.config = await requestJSON('/api/config');
  if (state.config) {
    state.graphLimit = clampInt(state.graphLimit || state.config.graph_limit || 160, 24, 1000, 160);
    el.graphLimitInput.value = String(state.graphLimit);
  }
}

async function refreshSessions() {
  const previousSessionId = state.selectedSessionId;
  state.sessions = await requestJSON('/api/chat/sessions');
  if (!state.sessions.some((session) => session.session_id === state.selectedSessionId) && state.sessions.length > 0) {
    state.selectedSessionId = state.sessions[0].session_id;
    el.chatSessionInput.value = state.selectedSessionId;
    localStorage.setItem('semantic_ants.session', state.selectedSessionId);
  }
  if (state.selectedSessionId !== previousSessionId) {
    clearChatWorkspace();
  }
}

async function refreshGraph() {
  const query = el.graphQueryInput.value.trim();
  state.graphQuery = query;
  state.graphLimit = clampInt(el.graphLimitInput.value, 24, 1000, 160);
  const params = new URLSearchParams();
  if (query) params.set('query', query);
  params.set('limit', String(state.graphLimit));
  if (state.lastResult?.result_id) {
    params.set('result_id', state.lastResult.result_id);
  }
  state.graph = await requestJSON(`/api/graph?${params.toString()}`);
  renderGraphPanel();
}

async function refreshJobs() {
  state.jobs = await requestJSON('/api/jobs');
  renderJobs();
  return state.jobs;
}

async function sendChatMessage(event) {
  event.preventDefault();
  const text = el.chatInput.value.trim();
  if (!text || state.busy) return;
  state.busy = true;
  renderTopbar();
  try {
    const payload = {
      text,
      session_id: sanitizeSessionId(el.chatSessionInput.value || state.selectedSessionId),
      backpack_limit: 48,
      include_graph: false,
      include_layers: false,
      include_trace: false,
    };
    const response = await requestJSON('/api/chat/message', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    state.lastResult = response.result || null;
    state.backpack = response.backpack || response;
    state.graph = state.graph || null;
    state.activeDetailKind = 'result';
    state.nodeDetail = null;
    state.selectedNodeId = null;
    state.selectedEdgeId = null;
    state.selectedSessionId = payload.session_id;
    el.chatSessionInput.value = state.selectedSessionId;
    localStorage.setItem('semantic_ants.session', state.selectedSessionId);
    el.chatInput.value = '';
    mergeResultIntoSessions(payload.session_id, state.lastResult);
    renderAll();
    scrollChatToEnd();
    const resultId = state.lastResult?.result_id || null;
    if (resultId) {
      const requestId = ++state.visualRequestId;
      loadChatBackpack({
        sessionId: payload.session_id,
        resultId,
        requestId,
      }).catch((error) => console.warn('lazy backpack failed', error));
    }
    refreshSessions()
      .then(() => {
        renderChat();
        scrollChatToEnd();
      })
      .catch(() => undefined);
    refreshConfig().then(renderTopbar).catch(() => undefined);
  } finally {
    state.busy = false;
    renderTopbar();
  }
}

async function startTraining() {
  const text = el.trainInput.value.trim();
  if (!text || state.busy) return;
  state.busy = true;
  renderTopbar();
  try {
    const payload = {
      text,
      session_id: sanitizeSessionId(el.trainSessionInput.value || state.trainSessionId),
      epochs: clampInt(el.trainEpochsInput.value, 1, 100, 1),
    };
    const job = await requestJSON('/api/train', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    state.jobs = [job, ...state.jobs.filter((item) => item.job_id !== job.job_id)];
    renderJobs();
    pollJobs();
  } finally {
    state.busy = false;
    renderTopbar();
  }
}

async function drillBackpackDown() {
  const nodeId = state.selectedNodeId;
  if (!nodeId || state.busy) return;
  state.busy = true;
  renderTopbar();
  try {
    const response = await requestJSON('/api/chat/drill-down', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sanitizeSessionId(el.chatSessionInput.value || state.selectedSessionId),
        node_id: nodeId,
      }),
    });
    state.backpack = response;
    state.graph = state.graph || null;
    state.activeDetailKind = 'none';
    await loadChatBackpack({ sessionId: sanitizeSessionId(el.chatSessionInput.value || state.selectedSessionId) });
    renderAll();
  } finally {
    state.busy = false;
    renderTopbar();
  }
}

async function drillBackpackUp() {
  if (state.busy) return;
  state.busy = true;
  renderTopbar();
  try {
    const response = await requestJSON('/api/chat/drill-up', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sanitizeSessionId(el.chatSessionInput.value || state.selectedSessionId),
      }),
    });
    state.backpack = response;
    state.graph = state.graph || null;
    state.activeDetailKind = 'none';
    await loadChatBackpack({ sessionId: sanitizeSessionId(el.chatSessionInput.value || state.selectedSessionId) });
    renderAll();
  } finally {
    state.busy = false;
    renderTopbar();
  }
}

function mergeResultIntoSessions(sessionId, result) {
  if (!result) return;
  const id = sanitizeSessionId(sessionId || result.session_id || state.selectedSessionId);
  let session = state.sessions.find((item) => item.session_id === id);
  if (!session) {
    session = { session_id: id, turns: [], turn_count: 0, updated_at: result.created_at || Date.now() / 1000 };
    state.sessions = [session, ...state.sessions];
  }
  const turns = Array.isArray(session.turns) ? session.turns.filter((turn) => turn.result_id !== result.result_id) : [];
  turns.push({
    role: 'user',
    text: result.input_text || '',
    created_at: result.created_at,
    result_id: result.result_id,
    kind: 'message',
  });
  turns.push({
    role: 'assistant',
    text: result.response || '',
    created_at: result.created_at,
    result_id: result.result_id,
    kind: 'message',
    source: result.response_source,
  });
  session.turns = turns;
  session.turn_count = turns.length;
  session.updated_at = result.created_at || session.updated_at || Date.now() / 1000;
}

async function loadChatBackpack({ sessionId, resultId = null, requestId = null } = {}) {
  const params = new URLSearchParams();
  params.set('session_id', sanitizeSessionId(sessionId || state.selectedSessionId));
  params.set('backpack_limit', '48');
  let response;
  if (resultId) {
    params.set('graph_limit', String(state.graphLimit || 120));
    params.set('layers', 'false');
    response = await requestJSON(`/api/chat/results/${encodeURIComponent(resultId)}/visuals?${params.toString()}`);
    if (requestId != null && requestId !== state.visualRequestId) return null;
    if (state.lastResult?.result_id && state.lastResult.result_id !== resultId) return null;
    state.backpack = response.backpack || response;
    state.graph = response.graph || state.graph || null;
  } else {
    response = await requestJSON(`/api/chat/backpack?${params.toString()}`);
    state.backpack = response;
  }
  renderGraphPanel();
  renderDetail();
  return response;
}

function renderAll() {
  renderTopbar();
  renderChat();
  renderGraphPanel();
  renderJobs();
  renderDetail();
}

function renderTopbar() {
  const chips = [];
  if (state.config) {
    chips.push(chip(`tokens ${state.config.tokens ?? 0}`));
    chips.push(chip(`edges ${state.config.edges ?? 0}`));
    chips.push(chip(`sessions ${state.config.sessions ?? 0}`));
  }
  chips.push(chip(`view ${state.view}`));
  if (state.busy) chips.push(chip('busy', 'tag-warning', true));
  if (state.lastResult?.response_source) chips.push(chip(state.lastResult.response_source, 'tag-accent'));
  el.statusStrip.innerHTML = chips.join('');
}

function renderChat() {
  const session = state.sessions.find((item) => item.session_id === state.selectedSessionId);
  const turns = session?.turns || [];
  if (!turns.length) {
    el.chatFeed.innerHTML = '<div class="empty-state">The selected session is empty.</div>';
    return;
  }
  el.chatFeed.innerHTML = turns
    .map((turn) => {
      const bubbleClass = turn.role === 'assistant' ? 'bubble-assistant' : turn.role === 'training' ? 'bubble-training' : 'bubble-user';
      const tags = [];
      if (turn.result_id) tags.push(`<span class="tag tag-accent">${escapeHtml(shortId(turn.result_id))}</span>`);
      if (turn.kind) tags.push(`<span class="tag">${escapeHtml(turn.kind)}</span>`);
      return `
        <article class="bubble ${bubbleClass}">
          <div class="bubble-meta">
            <strong>${escapeHtml(turn.role || 'unknown')}</strong>
            <span class="muted">${escapeHtml(formatTime(turn.created_at))}</span>
          </div>
          <p>${escapeHtml(turn.text || '')}</p>
          ${tags.length ? `<div class="bubble-tags">${tags.join('')}</div>` : ''}
        </article>
      `;
    })
    .join('');
}

function renderGraphPanel() {
  const graphData = state.view === 'chat' ? extractGraphData(state.backpack) : state.graph;
  if (!graphData) {
    destroyGraphInstance(el.backpackGraph);
    destroyGraphInstance(el.globalGraph);
    if (state.view === 'chat') {
      el.backpackStats.textContent = '0 nodes';
    }
    return;
  }
  if (state.view === 'chat') {
    renderGraphCanvas(el.backpackGraph, graphData, {
      selectedNodeId: state.selectedNodeId,
      selectedEdgeId: state.selectedEdgeId,
      kind: 'backpack',
    });
    el.backpackStats.textContent = backpackStatsLabel(state.backpack, graphData);
    return;
  }
  renderGraphCanvas(el.globalGraph, graphData, {
    selectedNodeId: state.selectedNodeId,
    selectedEdgeId: state.selectedEdgeId,
    kind: 'graph',
  });
}

function backpackStatsLabel(backpack, graphData) {
  const nodes = graphData?.nodes?.length || 0;
  const edges = graphData?.edges?.length || 0;
  const depth = Number(backpack?.current_depth ?? graphData?.current_depth ?? 0);
  const total = Number(backpack?.total_depth_layers ?? graphData?.total_depth_layers ?? 0);
  const focus = backpack?.active_focus_label || graphData?.active_focus_label || '0 nodes';
  return `${nodes} nodes | ${edges} edges | depth ${depth}/${total} | ${focus}`;
}

function clearChatWorkspace() {
  state.backpack = null;
  state.lastResult = null;
  state.selectedNodeId = null;
  state.selectedEdgeId = null;
  state.nodeDetail = null;
  state.activeDetailKind = 'none';
  destroyGraphInstance(el.backpackGraph);
}

function extractGraphData(payload) {
  if (!payload) return null;
  if (payload.graph_data) return payload.graph_data;
  if (payload.nodes && payload.edges) return payload;
  return null;
}

function renderGraphCanvas(container, graph, { selectedNodeId, selectedEdgeId, kind }) {
  if (!container || !graph) return;
  if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
    destroyGraphInstance(container);
    container.innerHTML = '';
    return;
  }

  const graphId = String(graph.graph_id || `${kind}:${graph.nodes.length}:${graph.edges.length}:${graph.seed_ids?.join('|') || ''}`);
  const displayLabelThreshold = 120;
  const displayLabels = graph.nodes.length <= displayLabelThreshold;
  const elements = buildCytoscapeElements(graph, { displayLabels, selectedNodeId, selectedEdgeId });
  const existing = state.graphInstances.get(container.id);
  if (existing && existing.graphId === graphId) {
    syncCytoscapeInstance(existing.cy, elements, { selectedNodeId, selectedEdgeId });
    return;
  }

  destroyGraphInstance(container);
  container.innerHTML = '';

  const cy = cytoscape({
    container,
    elements,
    layout: { name: 'preset', fit: false, animate: false },
    wheelSensitivity: 0.2,
    minZoom: 0.2,
    maxZoom: 4,
    selectionType: 'single',
    boxSelectionEnabled: false,
    autoungrabify: false,
    style: cytoscapeStyles(),
  });

  cy.ready(() => {
    cy.fit(undefined, 40);
    syncGraphSelection(cy, selectedNodeId, selectedEdgeId);
  });

  cy.on('tap', 'node', async (event) => {
    const node = event.target;
    state.selectedNodeId = node.id();
    state.selectedEdgeId = null;
    await loadNodeDetail(node.id());
    syncGraphSelection(cy, state.selectedNodeId, state.selectedEdgeId);
  });

  cy.on('tap', 'edge', (event) => {
    const edge = event.target;
    state.selectedNodeId = null;
    state.selectedEdgeId = edge.id();
    state.nodeDetail = { edge: edge.data() };
    state.activeDetailKind = 'edge';
    renderDetail();
    syncGraphSelection(cy, state.selectedNodeId, state.selectedEdgeId);
  });

  cy.on('tap', (event) => {
    if (event.target === cy) {
      state.selectedNodeId = null;
      state.selectedEdgeId = null;
      state.activeDetailKind = state.lastResult ? 'result' : 'none';
      renderDetail();
      syncGraphSelection(cy, state.selectedNodeId, state.selectedEdgeId);
    }
  });

  state.graphInstances.set(container.id, { cy, graphId });
}

function buildCytoscapeElements(graph, { displayLabels, selectedNodeId, selectedEdgeId }) {
  const nodes = graph.nodes.map((node) => {
    const isHyper = String(node.type || '').toLowerCase() === 'hypernode' || String(node.id || '').startsWith('hyper:');
    const label = String(node.label || node.token || node.id || '');
    return {
      data: {
        id: String(node.id || ''),
        label,
        display_label: displayLabels ? truncateLabel(label, isHyper ? 22 : 16) : '',
        type: isHyper ? 'hypernode' : 'token',
        count: Number(node.count || 0),
        score: Number(node.score || 0),
        relation: node.relation || '',
        active: Boolean(node.active),
        selected: String(node.id || '') === String(selectedNodeId || ''),
        shape: isHyper ? 'round-rectangle' : 'ellipse',
        radius: Number(node.radius || (isHyper ? 40 : 18)),
        width: Number(node.width || (isHyper ? 190 : 36)),
        height: Number(node.height || (isHyper ? 52 : 36)),
      },
      position: {
        x: Number(node.x || 0),
        y: Number(node.y || 0),
      },
      selectable: true,
      grabbable: true,
      classes: [
        isHyper ? 'hypernode' : 'token',
        node.active ? 'active' : '',
        String(node.id || '') === String(selectedNodeId || '') ? 'selected' : '',
      ]
        .filter(Boolean)
        .join(' '),
    };
  });

  const nodeSet = new Set(nodes.map((node) => node.data.id));
  const edges = graph.edges
    .filter((edge) => nodeSet.has(String(edge.source || '')) && nodeSet.has(String(edge.target || '')))
    .map((edge) => ({
      data: {
        id: String(edge.id || `${edge.source}|${edge.type || edge.relation || 'edge'}|${edge.target}`),
        source: String(edge.source || ''),
        target: String(edge.target || ''),
        relation: String(edge.relation || edge.type || 'next'),
        type: String(edge.type || edge.relation || 'transition_edge'),
        weight: Number(edge.weight || 0),
        active: Boolean(edge.active),
        selected: String(edge.id || '') === String(selectedEdgeId || ''),
      },
      classes: [
        String(edge.id || '') === String(selectedEdgeId || '') ? 'selected' : '',
        edge.active ? 'active' : '',
        String(edge.type || edge.relation || '') === 'hierarchical_edge' ? 'hierarchical-edge' : '',
        String(edge.type || edge.relation || '') === 'transition_edge' ? 'transition-edge' : '',
      ]
        .filter(Boolean)
        .join(' '),
    }));

  return [...nodes, ...edges];
}

function syncCytoscapeInstance(cy, elements, { selectedNodeId, selectedEdgeId }) {
  const existingIds = new Set(cy.elements().map((ele) => ele.id()));
  const nextIds = new Set(elements.map((ele) => ele.data.id));
  if (existingIds.size !== nextIds.size || [...existingIds].some((id) => !nextIds.has(id))) {
    cy.elements().remove();
    cy.add(elements);
    cy.layout({ name: 'preset', fit: false, animate: false }).run();
    cy.fit(undefined, 40);
  }
  syncGraphSelection(cy, selectedNodeId, selectedEdgeId);
}

function syncGraphSelection(cy, selectedNodeId, selectedEdgeId) {
  cy.nodes().forEach((node) => {
    node.toggleClass('selected', node.id() === String(selectedNodeId || ''));
  });
  cy.edges().forEach((edge) => {
    edge.toggleClass('selected', edge.id() === String(selectedEdgeId || ''));
  });
}

function cytoscapeStyles() {
  return [
    {
      selector: 'node',
      style: {
        'background-color': 'rgba(120, 241, 209, 0.18)',
        'border-color': 'rgba(120, 241, 209, 0.72)',
        'border-width': 1.8,
        width: 'data(width)',
        height: 'data(height)',
        shape: 'data(shape)',
        label: 'data(display_label)',
        color: '#f5f8ff',
        'font-family': 'var(--font-body)',
        'font-size': 10,
        'text-wrap': 'ellipsis',
        'text-max-width': 120,
        'text-valign': 'center',
        'text-halign': 'center',
        'text-outline-color': '#07111f',
        'text-outline-width': 2,
        'overlay-opacity': 0,
      },
    },
    {
      selector: 'node.hypernode',
      style: {
        'background-color': 'rgba(255, 207, 116, 0.16)',
        'border-color': 'rgba(255, 207, 116, 0.76)',
        'border-width': 2.4,
        'font-size': 11,
      },
    },
    {
      selector: 'node.active',
      style: {
        'background-color': 'rgba(120, 241, 209, 0.28)',
        'border-color': 'rgba(255, 255, 255, 0.95)',
      },
    },
    {
      selector: 'node.selected',
      style: {
        'border-color': 'rgba(255, 255, 255, 0.95)',
        'border-width': 3,
        'shadow-blur': 18,
        'shadow-color': 'rgba(120, 241, 209, 0.32)',
        'shadow-opacity': 0.9,
        'shadow-offset-x': 0,
        'shadow-offset-y': 0,
      },
    },
    {
      selector: 'edge',
      style: {
        width: 'mapData(weight, 0, 8, 0.8, 4.8)',
        'line-color': 'rgba(162, 189, 255, 0.22)',
        'target-arrow-color': 'rgba(162, 189, 255, 0.22)',
        'target-arrow-shape': 'triangle',
        'curve-style': 'bezier',
        opacity: 0.9,
      },
    },
    {
      selector: 'edge.hierarchical-edge',
      style: {
        'line-style': 'dashed',
        'line-color': 'rgba(255, 207, 116, 0.28)',
        'target-arrow-color': 'rgba(255, 207, 116, 0.28)',
      },
    },
    {
      selector: 'edge.transition-edge',
      style: {
        'line-color': 'rgba(120, 241, 209, 0.22)',
        'target-arrow-color': 'rgba(120, 241, 209, 0.22)',
      },
    },
    {
      selector: 'edge.active',
      style: {
        'line-color': 'rgba(120, 241, 209, 0.65)',
        'target-arrow-color': 'rgba(120, 241, 209, 0.65)',
        width: 4.2,
      },
    },
    {
      selector: 'edge.selected',
      style: {
        'line-color': 'rgba(255, 255, 255, 0.95)',
        'target-arrow-color': 'rgba(255, 255, 255, 0.95)',
        width: 5,
      },
    },
  ];
}

function destroyGraphInstance(container) {
  if (!container) return;
  const existing = state.graphInstances.get(container.id);
  if (existing) {
    existing.cy.destroy();
    state.graphInstances.delete(container.id);
  }
}

async function loadNodeDetail(nodeId) {
  try {
    const detail = await requestJSON(`/api/node/${encodeURIComponent(nodeId)}`);
    state.nodeDetail = detail;
    state.activeDetailKind = 'node';
  } catch (error) {
    state.nodeDetail = {
      node: { id: nodeId, label: nodeId, type: 'token' },
      neighbors: [],
      examples: [],
      error: String(error),
    };
    state.activeDetailKind = 'node';
  }
  renderDetail();
}

function renderDetail() {
  if (state.activeDetailKind === 'node' && state.nodeDetail) {
    el.chatDetail.innerHTML = renderNodeDetailHtml(state.nodeDetail);
    el.graphDetail.innerHTML = renderNodeDetailHtml(state.nodeDetail);
    return;
  }
  if (state.activeDetailKind === 'edge' && state.nodeDetail) {
    el.chatDetail.innerHTML = renderEdgeDetailHtml(state.nodeDetail);
    el.graphDetail.innerHTML = renderEdgeDetailHtml(state.nodeDetail);
    return;
  }
  if (state.lastResult) {
    const html = renderResultDetailHtml(state.lastResult);
    el.chatDetail.innerHTML = html;
    el.graphDetail.innerHTML = html;
    return;
  }
  const empty = '<div class="empty-state">Select a node in the graph.</div>';
  el.chatDetail.innerHTML = empty;
  el.graphDetail.innerHTML = empty;
}

function renderJobs() {
  const jobs = state.jobs || [];
  el.jobsCount.textContent = `${jobs.length} jobs`;
  if (!jobs.length) {
    el.jobsList.innerHTML = '<div class="empty-state">No jobs running.</div>';
    return;
  }
  el.jobsList.innerHTML = jobs
    .map((job) => {
      const statusClass = job.status || 'queued';
      const resultHtml = job.result ? `<pre>${escapeHtml(JSON.stringify(job.result, null, 2))}</pre>` : '';
      const errorHtml = job.error ? `<div class="tag tag-danger">${escapeHtml(job.error)}</div>` : '';
      return `
        <article class="job">
          <div class="job-head">
            <strong>${escapeHtml(job.kind || 'job')} | ${escapeHtml(shortId(job.job_id || ''))}</strong>
            <span class="job-status ${escapeHtml(statusClass)}">${escapeHtml(statusClass)}</span>
          </div>
          <div class="muted">${escapeHtml(formatTime(job.created_at))}</div>
          ${errorHtml}
          ${resultHtml}
        </article>
      `;
    })
    .join('');
}

function renderNodeDetailHtml(detail) {
  const node = detail.node || {};
  const neighbors = Array.isArray(detail.neighbors) ? detail.neighbors : [];
  const examples = Array.isArray(detail.examples) ? detail.examples : [];
  const hierarchy = Array.isArray(node.hierarchy) ? node.hierarchy.join(' / ') : '';
  const extraRows = [];
  if (node.parent) extraRows.push(['Parent', node.parent]);
  if (node.depth != null) extraRows.push(['Depth', String(node.depth)]);
  if (hierarchy) extraRows.push(['Hierarchy', hierarchy]);
  return `
    <div class="detail-card">
      <div class="detail-title">
        <strong>${escapeHtml(node.label || node.id || 'node')}</strong>
        <span class="tag">${escapeHtml(node.type || 'node')}</span>
      </div>
      <div class="detail-grid">
        <div class="muted">ID</div>
        <div>${escapeHtml(node.id || '')}</div>
        <div class="muted">Count</div>
        <div>${escapeHtml(String(node.count ?? 0))}</div>
        <div class="muted">Vector</div>
        <div>${escapeHtml(String(node.vector_norm ?? '0'))}</div>
        ${extraRows
          .map(([label, value]) => `
            <div class="muted">${escapeHtml(label)}</div>
            <div>${escapeHtml(value)}</div>
          `)
          .join('')}
      </div>
    </div>
    <div class="detail-card">
      <strong>Neighbors</strong>
      <div class="detail-list">
        ${neighbors
          .map((item) => {
            const neighbor = item.node || {};
            const edge = item.edge || {};
            return `
              <div class="detail-item">
                <div class="turn-head">
                  <strong>${escapeHtml(neighbor.label || neighbor.id || 'node')}</strong>
                  <span class="tag">${escapeHtml(edge.relation || '')}</span>
                </div>
                <div class="muted">${escapeHtml(String(edge.weight ?? 0))}</div>
              </div>
            `;
          })
          .join('')}
      </div>
    </div>
    <div class="detail-card">
      <strong>Examples</strong>
      <div class="detail-list">
        ${examples
          .map((item) => `<div class="detail-item"><div>${escapeHtml(item.text || '')}</div></div>`)
          .join('')}
      </div>
    </div>
  `;
}

function renderEdgeDetailHtml(detail) {
  const edge = detail.edge || detail.node || detail;
  return `
    <div class="detail-card">
      <div class="detail-title">
        <strong>${escapeHtml(edge.type || edge.relation || 'edge')}</strong>
        <span class="tag">${escapeHtml(shortId(edge.id || ''))}</span>
      </div>
      <div class="detail-grid">
        <div class="muted">Source</div>
        <div>${escapeHtml(edge.source || '')}</div>
        <div class="muted">Target</div>
        <div>${escapeHtml(edge.target || '')}</div>
        <div class="muted">Weight</div>
        <div>${escapeHtml(String(edge.weight ?? 0))}</div>
      </div>
    </div>
  `;
}

function renderResultDetailHtml(result) {
  const topTokens = Array.isArray(result.top_tokens) ? result.top_tokens : [];
  return `
    <div class="detail-card">
      <div class="detail-title">
        <strong>Latest response</strong>
        <span class="tag tag-accent">${escapeHtml(result.response_source || 'result')}</span>
      </div>
      <div>${escapeHtml(result.response || '')}</div>
      <div class="muted">${escapeHtml(result.summary || '')}</div>
    </div>
    <div class="detail-card">
      <strong>Top tokens</strong>
      <div class="detail-list">
        ${topTokens
          .map((item) => `<div class="detail-item"><div>${escapeHtml(item.label || item.token || '')}</div><div class="muted">${escapeHtml(String(item.score ?? 0))}</div></div>`)
          .join('')}
      </div>
    </div>
  `;
}

function scrollChatToEnd() {
  el.chatFeed.scrollTop = el.chatFeed.scrollHeight;
}

function startJobsPolling() {
  if (jobTimer) return;
  jobTimer = window.setInterval(async () => {
    const hadActive = state.jobs.some((job) => ['queued', 'running'].includes(job.status));
    const jobs = await refreshJobs();
    const hasActive = jobs.some((job) => ['queued', 'running'].includes(job.status));
    if (hadActive && !hasActive) {
      await Promise.all([refreshConfig(), refreshGraph()]);
    }
    renderTopbar();
  }, 1200);
}

function pollJobs() {
  refreshJobs()
    .then(() => {
      renderTopbar();
    })
    .catch(() => undefined);
}

function chip(text, extraClass = '', withSpinner = false) {
  return `<span class="chip ${extraClass}">${withSpinner ? '<span class="spinner"></span>' : ''}${escapeHtml(text)}</span>`;
}

function sanitizeSessionId(value) {
  return String(value || 'default').trim() || 'default';
}

function clampInt(value, min, max, fallback) {
  const parsed = Number.parseInt(String(value), 10);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function shortId(value) {
  return String(value || '').slice(0, 8);
}

function truncateLabel(value, maxLength) {
  const text = String(value || '');
  if (text.length <= maxLength) return text;
  return `${text.slice(0, Math.max(maxLength - 3, 1))}...`;
}

function formatTime(ts) {
  if (!ts) return '—';
  const date = new Date(Number(ts) * 1000 || Number(ts));
  return date.toLocaleString([], {
    hour: '2-digit',
    minute: '2-digit',
    day: '2-digit',
    month: 'short',
  });
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

async function requestJSON(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  if (response.status === 204) return null;
  return await response.json();
}

function renderError(error) {
  document.body.innerHTML = `
    <div style="padding:24px;color:#fff;font-family:var(--font-body);">
      <h1>Startup error</h1>
      <pre>${escapeHtml(String(error?.stack || error))}</pre>
    </div>
  `;
}
