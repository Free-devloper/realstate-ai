const elements = {
    propertiesGrid: document.getElementById('properties-grid'),
    propertiesPagination: document.getElementById('properties-pagination'),
    recommendationsGrid: document.getElementById('recommendations-grid'),
    agentsGrid: document.getElementById('agents-grid'),
    heroSearchForm: document.getElementById('hero-search-form'),
    applyFiltersBtn: document.getElementById('apply-filters'),
    clearFiltersBtn: document.getElementById('clear-filters'),
    authModal: document.getElementById('auth-modal'),
    propertyModal: document.getElementById('property-modal'),
    authTabs: document.querySelectorAll('.tab-btn'),
    loginForm: document.getElementById('login-form'),
    registerForm: document.getElementById('register-form'),
    authStatus: document.getElementById('auth-status'),
    loginTrigger: document.getElementById('login-trigger'),
    registerTrigger: document.getElementById('register-trigger'),
    logoutBtn: document.getElementById('logout-btn'),
    feedback: document.getElementById('global-feedback'),
    recommendationSummary: document.getElementById('recommendation-summary'),
    propertyDetail: document.getElementById('property-detail'),
    scrollToTop: document.getElementById('scroll-to-top'),
    aiAgentGrid: document.getElementById('ai-agent-grid'),
    aiRuntimeStatus: document.getElementById('ai-runtime-status'),
    aiAgentSelect: document.getElementById('ai-agent-select'),
    aiConversationSummary: document.getElementById('ai-conversation-summary'),
    aiConversationList: document.getElementById('ai-conversation-list'),
    aiNewConversationBtn: document.getElementById('ai-new-conversation-btn'),
    aiChatTitle: document.getElementById('ai-chat-title'),
    aiChatSubtitle: document.getElementById('ai-chat-subtitle'),
    aiQuickPrompts: document.getElementById('ai-quick-prompts'),
    aiChatMessages: document.getElementById('ai-chat-messages'),
    aiChatForm: document.getElementById('ai-chat-form'),
    aiChatInput: document.getElementById('ai-chat-input'),
    aiSendBtn: document.getElementById('ai-send-btn'),
    aiStreamingToggle: document.getElementById('ai-streaming-toggle'),
    metrics: {
        properties: document.getElementById('metric-properties'),
        available: document.getElementById('metric-available'),
        agents: document.getElementById('metric-agents'),
        average: document.getElementById('metric-average')
    }
};

const state = {
    currentPage: 1,
    totalPages: 1,
    currentFilters: {},
    isLoggedIn: false,
    currentUser: null,
    aiAgents: [],
    aiStatus: null,
    activeAiAgentId: null,
    aiConversations: [],
    activeConversationId: null,
    aiHistory: [],
    aiBusy: false,
    aiStreamingEnabled: true,
    propertyIndex: {}
};

const COVER_STYLES = new Set(['sunset-grove', 'skyline-blue', 'amber-shore', 'terracotta']);

document.addEventListener('DOMContentLoaded', initializeApp);

function hasAiConversationUI() {
    return Boolean(
        elements.aiConversationSummary
        && elements.aiConversationList
        && elements.aiNewConversationBtn
    );
}

function restoreAiPreferences() {
    try {
        const stored = localStorage.getItem('aiStreamingEnabled');
        state.aiStreamingEnabled = stored !== 'false';
    } catch (error) {
        state.aiStreamingEnabled = true;
    }

    if (elements.aiStreamingToggle) {
        elements.aiStreamingToggle.checked = state.aiStreamingEnabled;
    }
}

function persistAiPreferences() {
    try {
        localStorage.setItem('aiStreamingEnabled', String(state.aiStreamingEnabled));
    } catch (error) {
        // Ignore preference persistence failures.
    }
}

async function initializeApp() {
    restoreAiPreferences();
    setupEventListeners();
    restoreAuth();
    updateAuthUI();

    await Promise.all([
        loadOverview(),
        loadProperties(),
        loadRecommendations(),
        loadAgents(),
        loadAiAgents()
    ]);
}

async function fetchAPI(endpoint, options = {}) {
    const {
        params,
        body,
        headers = {},
        silent = false,
        ...fetchOptions
    } = options;

    const url = new URL(endpoint, window.location.origin);
    if (params) {
        Object.entries(params).forEach(([key, value]) => {
            if (value !== undefined && value !== null && value !== '') {
                url.searchParams.set(key, value);
            }
        });
    }

    const requestOptions = {
        ...fetchOptions,
        headers: {
            'Content-Type': 'application/json',
            ...headers
        }
    };

    if (body !== undefined) {
        requestOptions.body = typeof body === 'string' ? body : JSON.stringify(body);
    }

    const response = await fetch(url, requestOptions);
    const contentType = response.headers.get('content-type') || '';
    let payload;

    if (contentType.includes('application/json')) {
        payload = await response.json();
    } else {
        const text = await response.text();
        payload = text ? { error: text } : {};
    }

    if (!response.ok) {
        const message = payload.error || payload.message || 'Request failed.';
        if (!silent) {
            showToast(message, 'error');
        }
        throw new Error(message);
    }

    return payload;
}

async function loadOverview() {
    try {
        const data = await fetchAPI('/api/overview', { silent: true });
        elements.metrics.properties.textContent = formatCompactNumber(data.properties_count);
        elements.metrics.available.textContent = formatCompactNumber(data.available_count);
        elements.metrics.agents.textContent = formatCompactNumber(data.agents_count);
        elements.metrics.average.textContent = `$${formatCompactNumber(data.average_price)}`;
    } catch (error) {
        setFeedback('The dashboard metrics could not be loaded right now.', 'error');
    }
}

async function loadProperties(page = 1) {
    setGridState(
        elements.propertiesGrid,
        'Loading listings...',
        'Pulling current inventory and ranking fresh properties for you.',
        'loading'
    );

    try {
        const data = await fetchAPI('/api/properties', {
            params: {
                page,
                per_page: 6,
                ...state.currentFilters
            },
            silent: true
        });

        state.currentPage = data.page;
        state.totalPages = data.pages;

        renderProperties(elements.propertiesGrid, data.properties, {
            emptyTitle: 'No properties matched those filters.',
            emptyDescription: 'Try widening the budget, changing location, or clearing a few constraints.'
        });
        updatePagination(data.page, data.pages);
    } catch (error) {
        setGridState(
            elements.propertiesGrid,
            'Unable to load listings.',
            error.message,
            'error'
        );
        updatePagination(1, 0);
        setFeedback('We hit a problem loading listings. You can retry by changing filters or refreshing.', 'error');
    }
}

async function loadRecommendations() {
    setGridState(
        elements.recommendationsGrid,
        'Loading recommendations...',
        'Assembling AI-ranked homes with stronger fit signals.',
        'loading'
    );

    try {
        let data = null;

        if (state.currentUser) {
            const favoritesData = await fetchAPI('/api/recommendations/favorites', {
                params: { user_id: state.currentUser.id },
                silent: true
            }).catch(() => null);

            if (favoritesData && Array.isArray(favoritesData.recommendations) && favoritesData.recommendations.length > 0) {
                data = favoritesData;
                updateRecommendationSummary('These picks lean on the homes you saved as favorites.');
            }
        }

        if (!data) {
            data = await fetchAPI('/api/recommendations', {
                params: state.currentUser ? { user_id: state.currentUser.id } : {},
                silent: true
            });
            updateRecommendationSummary(
                state.currentUser
                    ? 'Your recent searches are helping shape these recommendations.'
                    : 'Sign in and save favorites to make this feed more personal.'
            );
        }

        renderProperties(elements.recommendationsGrid, data.recommendations, {
            emptyTitle: 'No recommendations yet.',
            emptyDescription: 'Try a search, save a favorite, or browse featured listings first.'
        });
    } catch (error) {
        setGridState(
            elements.recommendationsGrid,
            'Recommendations are unavailable right now.',
            error.message,
            'error'
        );
        updateRecommendationSummary('Recommendations will appear here after the service is reachable again.');
    }
}

async function loadAgents() {
    setGridState(
        elements.agentsGrid,
        'Loading agents...',
        'Fetching verified experts from the local roster.',
        'loading'
    );

    try {
        const agents = await fetchAPI('/api/agents', { silent: true });
        renderAgents(agents);
    } catch (error) {
        setGridState(
            elements.agentsGrid,
            'Unable to load agents.',
            error.message,
            'error'
        );
    }
}

async function loadAiAgents() {
    try {
        const payload = await fetchAPI('/api/ai/agents', { silent: true });
        state.aiAgents = Array.isArray(payload.agents) ? payload.agents : [];
        state.aiStatus = payload.status || null;

        if (state.aiAgents.length && !state.activeAiAgentId) {
            state.activeAiAgentId = state.aiAgents[0].id;
        }

        renderAiAgents();
        renderAiStatus();
        updateAiHeader();

        const activeAgent = getAiAgentById(state.activeAiAgentId) || state.aiAgents[0];
        renderQuickPrompts(activeAgent ? (activeAgent.quick_prompts || []) : []);

        if (state.currentUser) {
            await loadAiConversations({ createIfEmpty: false });
        } else if (!state.aiHistory.length && activeAgent) {
            state.aiHistory = [buildAiWelcomeEntry(activeAgent)];
            renderAiConversation();
        }
    } catch (error) {
        elements.aiAgentGrid.innerHTML = `
            <div class="state-card state-error">
                <h3>AI agents are unavailable.</h3>
                <p>${escapeHTML(error.message)}</p>
            </div>
        `;
        if (elements.aiAgentSelect) {
            elements.aiAgentSelect.innerHTML = '<option value="">Unavailable</option>';
            elements.aiAgentSelect.disabled = true;
        }
        elements.aiRuntimeStatus.textContent = 'AI agents offline';
        elements.aiRuntimeStatus.className = 'runtime-pill fallback';
        elements.aiChatMessages.innerHTML = '<div class="chat-empty">The AI concierge could not be loaded right now.</div>';
    }
}

function buildAiWelcomeEntry(agent) {
    return {
        role: 'assistant',
        content: buildAiWelcomeMessage(agent),
        source: 'welcome',
        agent_id: agent.id,
        agent_name: agent.name,
        recommended_properties: []
    };
}

function ensureGuestAiConversation(agent) {
    if (state.aiHistory.length) {
        renderAiConversations();
        renderAiConversation();
        return;
    }

    state.activeConversationId = null;
    state.aiConversations = [];
    state.aiHistory = [buildAiWelcomeEntry(agent)];
    renderAiConversations();
    renderAiConversation();
}

async function loadAiConversations(options = {}) {
    const { createIfEmpty = false } = options;
    const agent = getAiAgentById(state.activeAiAgentId) || state.aiAgents[0];
    if (!agent) {
        return;
    }

    if (!state.currentUser || !hasAiConversationUI()) {
        ensureGuestAiConversation(agent);
        return;
    }

    elements.aiConversationSummary.textContent = 'Loading shared AI conversations...';
    elements.aiConversationList.innerHTML = '<div class="chat-empty">Loading saved conversations...</div>';

    try {
        const payload = await fetchAPI(`/api/users/${state.currentUser.id}/ai/conversations`, { silent: true });
        state.aiConversations = Array.isArray(payload.conversations) ? payload.conversations : [];

        if (state.activeConversationId) {
            const stillActive = state.aiConversations.find((item) => item.id === state.activeConversationId);
            if (stillActive) {
                await openAiConversation(stillActive.id, { focus: false });
                return;
            }
        }

        if (state.aiConversations.length) {
            await openAiConversation(state.aiConversations[0].id, { focus: false });
            return;
        }

        if (createIfEmpty) {
            await createNewAiConversation({ focus: false });
            return;
        }

        state.activeConversationId = null;
        state.aiHistory = [buildAiWelcomeEntry(agent)];
        renderAiConversations();
        renderAiConversation();
    } catch (error) {
        elements.aiConversationSummary.textContent = error.message;
        elements.aiConversationList.innerHTML = '<div class="chat-empty">Saved conversations could not be loaded.</div>';
        ensureGuestAiConversation(agent);
    }
}

async function createNewAiConversation(options = {}) {
    const { focus = true } = options;
    const agent = getAiAgentById(state.activeAiAgentId) || state.aiAgents[0];
    if (!agent) {
        return;
    }

    if (!state.currentUser) {
        state.aiHistory = [buildAiWelcomeEntry(agent)];
        renderAiConversations();
        renderAiConversation();
        if (focus) {
            elements.aiChatInput.focus();
        }
        return;
    }

    const payload = await fetchAPI(`/api/users/${state.currentUser.id}/ai/conversations`, {
        method: 'POST',
        body: { agent_id: agent.id },
        silent: true
    });

    const conversation = payload.conversation;
    state.aiConversations = [conversation, ...state.aiConversations.filter((item) => item.id !== conversation.id)];
    state.activeConversationId = conversation.id;
    state.aiHistory = [buildAiWelcomeEntry(agent)];
    renderAiConversations();
    renderAiConversation();
    updateConversationSummary();

    if (focus) {
        elements.aiChatInput.focus();
    }
}

async function openAiConversation(conversationId, options = {}) {
    const { focus = false } = options;
    if (!state.currentUser) {
        return;
    }

    const payload = await fetchAPI(`/api/users/${state.currentUser.id}/ai/conversations/${conversationId}`, { silent: true });
    state.activeConversationId = payload.conversation.id;
    state.aiConversations = [
        payload.conversation,
        ...state.aiConversations.filter((item) => item.id !== payload.conversation.id)
    ];
    state.activeAiAgentId = payload.conversation.preferred_agent_id || payload.conversation.agent_id || state.activeAiAgentId;
    renderAiAgents();
    updateAiHeader();

    state.aiHistory = (payload.messages || []).map((message) => ({
        role: message.role,
        content: message.content,
        source: message.source,
        agent_id: message.agent_id,
        agent_name: message.agent_name,
        recommended_properties: message.recommended_properties || []
    }));

    if (!state.aiHistory.length) {
        const agent = getAiAgentById(state.activeAiAgentId) || state.aiAgents[0];
        state.aiHistory = agent ? [buildAiWelcomeEntry(agent)] : [];
    }

    renderAiConversations();
    renderAiConversation();
    updateConversationSummary();

    if (focus) {
        elements.aiChatInput.focus();
    }
}

function renderAiConversations() {
    if (!hasAiConversationUI()) {
        return;
    }

    if (!state.currentUser) {
        elements.aiConversationSummary.textContent = 'Sign in to save one shared thread where buyer, investment, and neighborhood specialists can all respond when relevant.';
        elements.aiConversationList.innerHTML = '<div class="chat-empty">Sign in to unlock saved conversations.</div>';
        return;
    }

    if (!state.aiConversations.length) {
        elements.aiConversationList.innerHTML = '<div class="chat-empty">No saved chats yet. Start one shared conversation with the button above.</div>';
        updateConversationSummary();
        return;
    }

    elements.aiConversationList.innerHTML = state.aiConversations.map((conversation) => {
        const preferredLabel = conversation.preferred_agent && conversation.preferred_agent.name
            ? `Preferred: ${conversation.preferred_agent.name}`
            : 'Shared AI chat';
        return `
            <button class="conversation-item ${conversation.id === state.activeConversationId ? 'active' : ''}" type="button" data-conversation-id="${conversation.id}">
                <span class="conversation-item-title">${escapeHTML(conversation.title || 'New conversation')}</span>
                <span class="conversation-item-meta">${escapeHTML(preferredLabel)} - ${escapeHTML(formatConversationTime(conversation.last_message_at || conversation.updated_at))}</span>
                <span class="conversation-item-preview">${escapeHTML(conversation.last_preview || 'No saved replies yet.')}</span>
            </button>
        `;
    }).join('');
    updateConversationSummary();
}

function updateConversationSummary() {
    if (!hasAiConversationUI()) {
        return;
    }

    if (!state.currentUser) {
        elements.aiConversationSummary.textContent = 'Sign in to save one shared thread where buyer, investment, and neighborhood specialists can all respond when relevant.';
        return;
    }

    const activeConversation = state.aiConversations.find((item) => item.id === state.activeConversationId);
    if (activeConversation) {
        const preferredName = activeConversation.preferred_agent && activeConversation.preferred_agent.name
            ? activeConversation.preferred_agent.name
            : 'the current specialist';
        elements.aiConversationSummary.textContent = `Working inside "${activeConversation.title}". ${preferredName} is the preferred specialist, but other agents can step in within the same thread.`;
        return;
    }

    elements.aiConversationSummary.textContent = 'Create a fresh shared conversation or reopen an earlier thread.';
}


function renderProperties(container, properties, options = {}) {
    const { emptyTitle = 'No results found.', emptyDescription = 'Try a different search.' } = options;

    if (!Array.isArray(properties) || properties.length === 0) {
        setGridState(container, emptyTitle, emptyDescription, 'empty');
        return;
    }

    cacheProperties(properties);

    container.innerHTML = properties.map((property) => {
        const status = escapeHTML(property.status || 'available');
        const coverStyle = getCoverStyle(property.cover_style);
        const reason = property.recommendation_reason
            ? `<p class="property-reason">${escapeHTML(property.recommendation_reason)}</p>`
            : '';
        const lookupKey = escapeHTML(getPropertyLookupKey(property));
        const sourceLabel = property.source_label || (property.is_external ? 'Live market feed' : 'Ranked by AI');

        return `
            <article class="property-card" data-property-ref="${lookupKey}" tabindex="0" role="button" aria-label="Open details for ${escapeHTML(property.title)}">
                <div class="property-image cover-${coverStyle}">
                    <div class="property-badges">
                        <span class="badge"><i class="fas fa-brain"></i> ${Number(property.ai_score || 0)}/100</span>
                        <span class="badge"><i class="fas fa-house"></i> ${escapeHTML(capitalizeLabel(property.property_type || 'house'))}</span>
                    </div>
                    <i class="fas fa-house-chimney"></i>
                </div>
                <div class="property-info">
                    <div class="property-price">$${formatPrice(property.price)}</div>
                    <h3 class="property-title">${escapeHTML(property.title)}</h3>
                    <div class="property-location">
                        <i class="fas fa-location-dot"></i>
                        <span>${escapeHTML(property.location)}</span>
                    </div>
                    <div class="property-features">
                        <span><i class="fas fa-bed"></i> ${escapeHTML(String(property.bedrooms))} Beds</span>
                        <span><i class="fas fa-bath"></i> ${escapeHTML(String(property.bathrooms))} Baths</span>
                        <span><i class="fas fa-ruler-combined"></i> ${formatArea(property.square_feet)} sq ft</span>
                    </div>
                    ${reason}
                    <div class="property-footer">
                        <span class="status-badge status-${status}">${escapeHTML(capitalizeLabel(status))}</span>
                        <span class="agent-chip"><i class="fas fa-sparkles"></i> ${escapeHTML(sourceLabel)}</span>
                    </div>
                </div>
            </article>
        `;
    }).join('');
}

function renderAgents(agents) {
    if (!Array.isArray(agents) || agents.length === 0) {
        setGridState(
            elements.agentsGrid,
            'No agents available yet.',
            'Add verified agents to showcase local expertise here.',
            'empty'
        );
        return;
    }

    elements.agentsGrid.innerHTML = agents.map((agent) => `
        <article class="agent-card">
            <div class="agent-top">
                <div class="agent-avatar">${escapeHTML(getInitials(agent.name))}</div>
                <div>
                    <h3 class="agent-name">${escapeHTML(agent.name)}</h3>
                    <p class="agent-company">${escapeHTML(agent.company || 'Independent advisor')}</p>
                </div>
            </div>
            <p class="agent-bio">${escapeHTML(agent.bio || 'Experienced in helping buyers compare value, location, and fit.')}</p>
            <div class="agent-contact">
                <span><i class="fas fa-phone"></i> ${escapeHTML(agent.phone || 'Not provided')}</span>
                <span><i class="fas fa-envelope"></i> ${escapeHTML(agent.email || 'Not provided')}</span>
            </div>
            <span class="agent-chip">
                <i class="fas fa-shield-heart"></i>
                ${agent.is_verified ? 'Verified professional' : 'Agent profile'}
            </span>
        </article>
    `).join('');
}

function renderAiAgents() {
    if (!state.aiAgents.length) {
        elements.aiAgentGrid.innerHTML = `
            <div class="state-card state-error">
                <h3>No AI agents configured.</h3>
                <p>Add an AI agent configuration to enable this section.</p>
            </div>
        `;
        if (elements.aiAgentSelect) {
            elements.aiAgentSelect.innerHTML = '<option value="">No agents available</option>';
            elements.aiAgentSelect.disabled = true;
        }
        return;
    }

    if (elements.aiAgentSelect) {
        elements.aiAgentSelect.disabled = false;
        elements.aiAgentSelect.innerHTML = state.aiAgents.map((agent) => `
            <option value="${escapeHTML(agent.id)}" ${agent.id === state.activeAiAgentId ? 'selected' : ''}>${escapeHTML(agent.name)}</option>
        `).join('');
    }

    elements.aiAgentGrid.innerHTML = state.aiAgents.map((agent) => `
        <article class="ai-agent-card ${agent.id === state.activeAiAgentId ? 'active' : ''}">
            <div class="ai-agent-top">
                <div>
                    <div class="ai-agent-name">${escapeHTML(agent.name)}</div>
                    <div class="ai-agent-role">${escapeHTML(agent.headline)}</div>
                </div>
                <span class="ai-agent-badge"><i class="fas fa-robot"></i></span>
            </div>
        </article>
    `).join('');
}

function renderAiStatus() {
    if (!state.aiStatus) {
        elements.aiRuntimeStatus.textContent = 'AI concierge';
        elements.aiRuntimeStatus.className = 'runtime-pill fallback';
        return;
    }

    if (state.aiStatus.remote_enabled) {
        elements.aiRuntimeStatus.textContent = 'AI concierge ready';
        elements.aiRuntimeStatus.className = 'runtime-pill live';
    } else {
        elements.aiRuntimeStatus.textContent = 'AI concierge ready';
        elements.aiRuntimeStatus.className = 'runtime-pill fallback';
    }
}

function getAiAgentById(agentId) {
    return state.aiAgents.find((item) => item.id === agentId) || null;
}

function updateAiHeader() {
    const agent = getAiAgentById(state.activeAiAgentId) || state.aiAgents[0] || null;
    if (!agent) {
        elements.aiChatTitle.textContent = 'Shared agent chat';
        elements.aiChatSubtitle.textContent = 'Ask about fit, value, neighborhoods, or the best next property to review.';
        return;
    }

    elements.aiChatTitle.textContent = 'Shared agent chat';
    elements.aiChatSubtitle.textContent = `${agent.name} is your preferred specialist right now. Other agents can jump in when the topic fits them better.`;
}

async function activateAiAgent(agentId, shouldFocus = false) {
    const agent = getAiAgentById(agentId);
    if (!agent) {
        return;
    }

    state.activeAiAgentId = agent.id;
    renderAiAgents();
    updateAiHeader();
    renderQuickPrompts(agent.quick_prompts || []);

    if (!state.aiHistory.length) {
        state.aiHistory = [buildAiWelcomeEntry(agent)];
        renderAiConversation();
    } else {
        renderAiConversation();
    }

    if (shouldFocus) {
        elements.aiChatInput.focus();
    }
}

function buildAiWelcomeMessage(agent) {
    return `Start with ${agent.name.toLowerCase()} as your preferred specialist. If your question shifts toward investing, neighborhoods, or buyer fit, another specialist can step in without leaving this chat.`;
}

function renderQuickPrompts(prompts) {
    if (!prompts.length) {
        elements.aiQuickPrompts.innerHTML = '';
        return;
    }

    elements.aiQuickPrompts.innerHTML = prompts.map((prompt) => `
        <button class="quick-prompt" type="button" data-quick-prompt="${escapeHTML(prompt)}">${escapeHTML(prompt)}</button>
    `).join('');
}

function getActiveAiAgentName() {
    const agent = getAiAgentById(state.activeAiAgentId);
    return agent ? agent.name : 'AI Concierge';
}

function renderAiConversation() {
    if (!state.aiHistory.length) {
        elements.aiChatMessages.innerHTML = '<div class="chat-empty">Choose a preferred agent and start a shared conversation.</div>';
        return;
    }

    state.aiHistory.forEach((entry) => {
        if (Array.isArray(entry.recommended_properties) && entry.recommended_properties.length) {
            cacheProperties(entry.recommended_properties);
        }
    });

    elements.aiChatMessages.innerHTML = state.aiHistory.map((entry) => {
        const recommendations = Array.isArray(entry.recommended_properties) && entry.recommended_properties.length
            ? `
                <div class="chat-recommendations">
                    ${entry.recommended_properties.map((property) => `
                        <div class="chat-recommendation">
                            <div>
                                <strong>${escapeHTML(property.title)}</strong>
                                <div>${escapeHTML(property.location)} | $${formatPrice(property.price)}</div>
                            </div>
                            <button type="button" data-open-property="${escapeHTML(getPropertyLookupKey(property))}">View</button>
                        </div>
                    `).join('')}
                </div>
            `
            : '';
        const notice = entry.notice
            ? `<div class="chat-notice">${escapeHTML(entry.notice)}</div>`
            : '';
        return `
            <div class="chat-bubble ${entry.role}${entry.isStreaming ? ' is-streaming' : ''}">
                <div class="chat-bubble-meta">
                    <span>${entry.role === 'assistant' ? escapeHTML(entry.agent_name || getActiveAiAgentName()) : 'You'}</span>
                </div>
                <div class="chat-body">${escapeHTML(entry.content)}</div>
                ${notice}
                ${recommendations}
            </div>
        `;
    }).join('');

    elements.aiChatMessages.scrollTop = elements.aiChatMessages.scrollHeight;
}

function setupEventListeners() {
    elements.heroSearchForm.addEventListener('submit', handleHeroSearch);
    elements.applyFiltersBtn.addEventListener('click', handleFilterApply);
    elements.clearFiltersBtn.addEventListener('click', clearFilters);
    elements.loginForm.addEventListener('submit', handleLogin);
    elements.registerForm.addEventListener('submit', handleRegister);
    elements.logoutBtn.addEventListener('click', logout);
    elements.aiChatForm.addEventListener('submit', handleAiChatSubmit);
    if (elements.aiStreamingToggle) {
        elements.aiStreamingToggle.addEventListener('change', () => {
            state.aiStreamingEnabled = elements.aiStreamingToggle.checked;
            persistAiPreferences();
            showToast(state.aiStreamingEnabled ? 'Streaming replies enabled.' : 'Streaming replies disabled.', 'info');
        });
    }
    if (elements.aiNewConversationBtn) {
        elements.aiNewConversationBtn.addEventListener('click', () => {
            createNewAiConversation({ focus: true }).catch((error) => {
                setFeedback(error.message, 'error');
            });
        });
    }
    if (elements.aiAgentSelect) {
        elements.aiAgentSelect.addEventListener('change', () => {
            activateAiAgent(elements.aiAgentSelect.value, true).catch((error) => {
                setFeedback(error.message, 'error');
            });
        });
    }
    elements.scrollToTop.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));

    document.querySelectorAll('.js-open-auth').forEach((button) => {
        button.addEventListener('click', () => openAuthModal(button.dataset.authTab || 'login'));
    });

    elements.authTabs.forEach((tab) => {
        tab.addEventListener('click', () => switchAuthTab(tab.dataset.tab));
    });

    document.querySelectorAll('.close').forEach((button) => {
        button.addEventListener('click', closeModals);
    });

    [elements.authModal, elements.propertyModal].forEach((modal) => {
        modal.addEventListener('click', (event) => {
            if (event.target === modal) {
                closeModals();
            }
        });
    });

    document.addEventListener('click', (event) => {
        const paginationButton = event.target.closest('.pagination-btn');
        if (paginationButton) {
            const page = Number(paginationButton.dataset.page);
            if (page && page !== state.currentPage) {
                loadProperties(page);
            }
            return;
        }

        const conversationButton = event.target.closest('[data-conversation-id]');
        if (conversationButton) {
            openAiConversation(Number(conversationButton.dataset.conversationId), { focus: true }).catch((error) => {
                setFeedback(error.message, 'error');
            });
            return;
        }

        const quickPromptButton = event.target.closest('[data-quick-prompt]');
        if (quickPromptButton) {
            elements.aiChatInput.value = quickPromptButton.dataset.quickPrompt;
            elements.aiChatInput.focus();
            return;
        }

        const openPropertyButton = event.target.closest('[data-open-property]');
        if (openPropertyButton) {
            showPropertyDetail(openPropertyButton.dataset.openProperty);
            return;
        }

        const propertyCard = event.target.closest('.property-card');
        if (propertyCard) {
            showPropertyDetail(propertyCard.dataset.propertyRef);
            return;
        }

        const favoriteButton = event.target.closest('[data-favorite-id]');
        if (favoriteButton) {
            addToFavorites(Number(favoriteButton.dataset.favoriteId));
        }
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            closeModals();
            return;
        }

        const propertyCard = event.target.closest ? event.target.closest('.property-card') : null;
        if (propertyCard && (event.key === 'Enter' || event.key === ' ')) {
            event.preventDefault();
            showPropertyDetail(propertyCard.dataset.propertyRef);
        }
    });

    window.addEventListener('scroll', () => {
        elements.scrollToTop.classList.toggle('visible', window.scrollY > 360);
    });
}

async function handleHeroSearch(event) {
    event.preventDefault();

    const filters = sanitizeFilters({
        query: document.getElementById('hero-location').value,
        location: document.getElementById('hero-location').value,
        min_price: document.getElementById('hero-min-price').value,
        max_price: document.getElementById('hero-max-price').value,
        property_type: document.getElementById('hero-property-type').value,
        min_bedrooms: document.getElementById('hero-min-bedrooms').value
    });

    state.currentFilters = { ...filters };
    syncFilterInputs(filters);
    await executeSmartSearch(filters, true);
    await loadProperties(1);
}

async function handleFilterApply() {
    const filters = sanitizeFilters({
        location: document.getElementById('filter-location').value,
        query: document.getElementById('filter-location').value,
        min_price: document.getElementById('filter-min-price').value,
        max_price: document.getElementById('filter-max-price').value,
        min_bedrooms: document.getElementById('filter-min-bedrooms').value,
        property_type: document.getElementById('filter-property-type').value,
        status: document.getElementById('filter-status').value
    });

    state.currentFilters = { ...filters };
    await executeSmartSearch(filters, false);
    await loadProperties(1);
}

function clearFilters() {
    state.currentFilters = {};

    [
        'hero-location',
        'hero-min-price',
        'hero-max-price',
        'hero-property-type',
        'hero-min-bedrooms',
        'filter-location',
        'filter-min-price',
        'filter-max-price',
        'filter-min-bedrooms',
        'filter-property-type',
        'filter-status'
    ].forEach((id) => {
        const input = document.getElementById(id);
        if (input) {
            input.value = '';
        }
    });

    setFeedback('Filters cleared. Showing the full listing feed again.', 'info');
    loadProperties(1);
    loadRecommendations();
}

async function executeSmartSearch(filters, scrollToResults) {
    if (!Object.keys(filters).length) {
        setFeedback('Add at least one search term or filter to run an AI search.', 'info');
        updateRecommendationSummary('Recommendations stay broad until you search or save favorites.');
        await loadRecommendations();
        return;
    }

    setFeedback('Running your AI search and refreshing the recommendation feed...', 'info');

    try {
        const data = await fetchAPI('/api/search', {
            method: 'POST',
            body: {
                ...filters,
                user_id: state.currentUser ? state.currentUser.id : undefined
            },
            silent: true
        });

        renderProperties(elements.recommendationsGrid, data.properties, {
            emptyTitle: 'Your search returned no close matches.',
            emptyDescription: 'Try widening the budget, changing the area, or reducing bedroom requirements.'
        });

        const searchSummary = data.agent_summary || buildSearchSummary(data.applied_filters, data.count);
        updateRecommendationSummary(searchSummary);
        const sourceLabel = data.source === 'live_market' ? 'live market' : 'local inventory';
        setFeedback(`Found ${data.count} ${data.count === 1 ? 'result' : 'results'} from the ${sourceLabel}.`, 'success');

        if (scrollToResults) {
            document.getElementById('recommendations').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    } catch (error) {
        setFeedback(error.message, 'error');
        setGridState(
            elements.recommendationsGrid,
            'Search could not be completed.',
            error.message,
            'error'
        );
    }
}

function buildAiChatRequestBody(message, priorHistory) {
    return {
        agent_id: state.activeAiAgentId,
        message,
        history: priorHistory,
        user_id: state.currentUser ? state.currentUser.id : undefined,
        conversation_id: state.currentUser ? state.activeConversationId : undefined
    };
}

function applyAiChatPayload(payload, assistantEntry) {
    state.aiStatus = payload.status || state.aiStatus;
    renderAiStatus();

    if (payload.preferred_agent && payload.preferred_agent.id) {
        state.activeAiAgentId = payload.preferred_agent.id;
        renderAiAgents();
        updateAiHeader();
    }

    if (payload.conversation) {
        state.activeConversationId = payload.conversation.id;
        state.aiConversations = [
            payload.conversation,
            ...state.aiConversations.filter((item) => item.id !== payload.conversation.id)
        ];
        renderAiConversations();
    }

    assistantEntry.content = payload.message || assistantEntry.content;
    assistantEntry.source = payload.source;
    assistantEntry.agent_id = payload.agent ? payload.agent.id : assistantEntry.agent_id;
    assistantEntry.agent_name = payload.agent ? payload.agent.name : assistantEntry.agent_name;
    assistantEntry.notice = buildAiNotice(payload);
    assistantEntry.recommended_properties = payload.recommended_properties || [];
    assistantEntry.isStreaming = false;
    renderAiConversation();

    if (payload.source === 'local_fallback' && payload.fallback_reason) {
        showToast('AI agent answered with the local fallback brain.', 'info');
    }
}

async function parseFailedResponseMessage(response) {
    const contentType = response.headers.get('content-type') || '';

    if (contentType.includes('application/json')) {
        const payload = await response.json();
        return payload.error || payload.message || 'Request failed.';
    }

    const text = await response.text();
    return text || 'Request failed.';
}

function parseStreamEventBlock(block) {
    const trimmed = String(block || '').trim();
    if (!trimmed) {
        return null;
    }

    let eventName = 'message';
    const dataLines = [];
    trimmed.split(/\r?\n/).forEach((line) => {
        if (line.startsWith('event:')) {
            eventName = line.slice(6).trim();
            return;
        }
        if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trim());
        }
    });

    if (!dataLines.length) {
        return null;
    }

    return {
        eventName,
        payload: JSON.parse(dataLines.join('\n'))
    };
}

async function streamAiChatResponse(body, assistantEntry) {
    const response = await fetch('/api/ai/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });

    if (!response.ok) {
        throw new Error(await parseFailedResponseMessage(response));
    }

    if (!response.body) {
        throw new Error('Streaming is not available in this browser.');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let donePayload = null;

    while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

        let boundary = buffer.indexOf('\n\n');
        while (boundary !== -1) {
            const block = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + 2);
            const parsed = parseStreamEventBlock(block);
            if (parsed) {
                const { eventName, payload } = parsed;
                if (eventName === 'meta') {
                    state.aiStatus = payload.status || state.aiStatus;
                    renderAiStatus();
                    if (payload.conversation) {
                        state.activeConversationId = payload.conversation.id;
                        state.aiConversations = [
                            payload.conversation,
                            ...state.aiConversations.filter((item) => item.id !== payload.conversation.id)
                        ];
                        renderAiConversations();
                    }
                    assistantEntry.source = payload.source || assistantEntry.source;
                    assistantEntry.notice = buildAiNotice(payload);
                } else if (eventName === 'delta') {
                    assistantEntry.content = assistantEntry.content
                        ? `${assistantEntry.content} ${payload.content}`
                        : String(payload.content || '');
                    assistantEntry.isStreaming = true;
                    renderAiConversation();
                } else if (eventName === 'done') {
                    donePayload = payload;
                    applyAiChatPayload(payload, assistantEntry);
                }
            }
            boundary = buffer.indexOf('\n\n');
        }

        if (done) {
            break;
        }
    }

    if (!donePayload) {
        assistantEntry.isStreaming = false;
        renderAiConversation();
        throw new Error('Streaming response ended unexpectedly.');
    }

    return donePayload;
}

async function handleAiChatSubmit(event) {
    event.preventDefault();
    if (state.aiBusy) {
        return;
    }

    const message = elements.aiChatInput.value.trim();
    if (!message) {
        showToast('Write a message for the AI agent first.', 'info');
        return;
    }

    const priorHistory = state.aiHistory.map((entry) => ({
        role: entry.role,
        content: entry.content,
        agent_id: entry.agent_id,
        agent_name: entry.agent_name
    }));
    const requestBody = buildAiChatRequestBody(message, priorHistory);
    state.aiHistory.push({ role: 'user', content: message });
    const assistantEntry = {
        role: 'assistant',
        content: state.aiStreamingEnabled ? '' : 'Thinking...',
        source: 'pending',
        agent_id: state.activeAiAgentId,
        agent_name: getActiveAiAgentName(),
        notice: '',
        recommended_properties: [],
        isStreaming: state.aiStreamingEnabled
    };
    state.aiHistory.push(assistantEntry);
    renderAiConversation();
    elements.aiChatInput.value = '';
    setAiBusy(true);

    try {
        if (state.aiStreamingEnabled) {
            await streamAiChatResponse(requestBody, assistantEntry);
        } else {
            const payload = await fetchAPI('/api/ai/chat', {
                method: 'POST',
                body: requestBody,
                silent: true
            });
            applyAiChatPayload(payload, assistantEntry);
        }
    } catch (error) {
        assistantEntry.content = `I hit a problem while responding: ${error.message}`;
        assistantEntry.source = 'error';
        assistantEntry.notice = '';
        assistantEntry.recommended_properties = [];
        assistantEntry.isStreaming = false;
        renderAiConversation();
        setFeedback(error.message, 'error');
    } finally {
        setAiBusy(false);
    }
}

function setAiBusy(isBusy) {
    state.aiBusy = isBusy;
    elements.aiSendBtn.disabled = isBusy;
    elements.aiChatInput.disabled = isBusy;
    if (elements.aiStreamingToggle) {
        elements.aiStreamingToggle.disabled = isBusy;
    }
    elements.aiSendBtn.innerHTML = isBusy
        ? '<i class="fas fa-spinner fa-spin"></i> Thinking'
        : '<i class="fas fa-paper-plane"></i> Send';
}

async function handleLogin(event) {
    event.preventDefault();

    try {
        const user = await fetchAPI('/api/users/login', {
            method: 'POST',
            body: {
                username: document.getElementById('login-username').value,
                password: document.getElementById('login-password').value
            }
        });

        state.currentUser = user;
        state.isLoggedIn = true;
        persistAuth();
        updateAuthUI();
        elements.loginForm.reset();
        closeModals();
        setFeedback(`Welcome back, ${user.username}. Your personalized recommendations are now live.`, 'success');
        await loadRecommendations();
        if (state.activeAiAgentId) {
            await loadAiConversations({ createIfEmpty: false });
        }
    } catch (error) {
        setFeedback(error.message, 'error');
    }
}

async function handleRegister(event) {
    event.preventDefault();

    try {
        await fetchAPI('/api/users/register', {
            method: 'POST',
            body: {
                username: document.getElementById('register-username').value,
                email: document.getElementById('register-email').value,
                password: document.getElementById('register-password').value,
                user_type: document.getElementById('register-user-type').value
            }
        });

        elements.registerForm.reset();
        switchAuthTab('login');
        showToast('Registration complete. You can log in now.', 'success');
        setFeedback('Account created successfully. Log in to unlock favorites and personalized picks.', 'success');
    } catch (error) {
        setFeedback(error.message, 'error');
    }
}

function restoreAuth() {
    try {
        const storedUser = localStorage.getItem('currentUser');
        if (!storedUser) {
            return;
        }

        const user = JSON.parse(storedUser);
        if (user && user.id) {
            state.currentUser = user;
            state.isLoggedIn = true;
        }
    } catch (error) {
        localStorage.removeItem('currentUser');
    }
}

function persistAuth() {
    if (state.isLoggedIn && state.currentUser) {
        localStorage.setItem('currentUser', JSON.stringify(state.currentUser));
    } else {
        localStorage.removeItem('currentUser');
    }
}

function updateAuthUI() {
    elements.logoutBtn.classList.toggle('hidden', !state.isLoggedIn);
    elements.loginTrigger.classList.toggle('hidden', state.isLoggedIn);
    elements.registerTrigger.classList.toggle('hidden', state.isLoggedIn);
    if (elements.aiNewConversationBtn) {
        elements.aiNewConversationBtn.disabled = false;
    }

    if (state.isLoggedIn && state.currentUser) {
        elements.authStatus.textContent = `Signed in as ${state.currentUser.username}. Save favorites to sharpen future recommendations.`;
    } else {
        elements.authStatus.textContent = 'Browse as a guest, or sign in to save favorites and unlock more tailored recommendations.';
    }
}

function openAuthModal(tab = 'login') {
    switchAuthTab(tab);
    elements.authModal.classList.add('active');
    elements.authModal.setAttribute('aria-hidden', 'false');
}

function switchAuthTab(tab) {
    elements.authTabs.forEach((button) => {
        button.classList.toggle('active', button.dataset.tab === tab);
    });

    const showLogin = tab === 'login';
    elements.loginForm.classList.toggle('hidden', !showLogin);
    elements.registerForm.classList.toggle('hidden', showLogin);
}

function closeModals() {
    elements.authModal.classList.remove('active');
    elements.propertyModal.classList.remove('active');
    elements.authModal.setAttribute('aria-hidden', 'true');
    elements.propertyModal.setAttribute('aria-hidden', 'true');
}

async function showPropertyDetail(propertyRef) {
    try {
        let property = getCachedProperty(propertyRef);
        if (!property) {
            const localPropertyId = findLocalPropertyId(propertyRef);
            if (localPropertyId) {
                property = await fetchAPI(`/api/properties/${localPropertyId}`, { silent: true });
                cacheProperties([property]);
            }
        }

        if (!property) {
            throw new Error('Property details are no longer in view. Run the search again to reopen this listing.');
        }

        const coverStyle = getCoverStyle(property.cover_style);
        const actionMarkup = property.can_favorite !== false && !property.is_external
            ? `
                <button class="btn btn-primary" type="button" data-favorite-id="${property.id}">
                    <i class="fas fa-heart"></i> Save to Favorites
                </button>
            `
            : property.listing_url
                ? `
                    <a class="btn btn-primary" href="${escapeHTML(property.listing_url)}" target="_blank" rel="noreferrer">
                        <i class="fas fa-arrow-up-right-from-square"></i> Open Listing
                    </a>
                `
                : '<span class="agent-chip"><i class="fas fa-bolt"></i> Live result</span>';

        elements.propertyDetail.innerHTML = `
            <div class="detail-header">
                <div class="detail-image cover-${coverStyle}">
                    <div class="property-badges">
                        <span class="badge"><i class="fas fa-brain"></i> ${Number(property.ai_score || 0)}/100</span>
                        <span class="badge"><i class="fas fa-location-dot"></i> ${escapeHTML(property.location)}</span>
                    </div>
                    <i class="fas fa-house-chimney-window"></i>
                </div>
                <div class="detail-info">
                    <div class="detail-price">$${formatPrice(property.price)}</div>
                    <h2 class="detail-title">${escapeHTML(property.title)}</h2>
                    <div class="detail-location">
                        <i class="fas fa-map-pin"></i>
                        <span>${escapeHTML(property.address)}, ${escapeHTML(property.location)}</span>
                    </div>
                    <div class="detail-features">
                        <div class="detail-feature"><i class="fas fa-bed"></i> ${escapeHTML(String(property.bedrooms))} Bedrooms</div>
                        <div class="detail-feature"><i class="fas fa-bath"></i> ${escapeHTML(String(property.bathrooms))} Bathrooms</div>
                        <div class="detail-feature"><i class="fas fa-ruler-combined"></i> ${formatArea(property.square_feet)} sq ft</div>
                        <div class="detail-feature"><i class="fas fa-building"></i> ${escapeHTML(capitalizeLabel(property.property_type || 'house'))}</div>
                    </div>
                    <div class="detail-score">
                        <strong>AI score:</strong> ${Number(property.ai_score || 0)}/100
                        <div class="detail-description">This score balances price fit, livability, freshness, and listing completeness.</div>
                    </div>
                    ${property.recommendation_reason ? `<div class="detail-reason">${escapeHTML(property.recommendation_reason)}</div>` : ''}
                    <p class="detail-description">${escapeHTML(property.description || 'No description provided for this listing yet.')}</p>
                    <div class="property-footer">
                        <span class="status-badge status-${escapeHTML(property.status || 'available')}">${escapeHTML(capitalizeLabel(property.status || 'available'))}</span>
                        ${actionMarkup}
                    </div>
                </div>
            </div>
        `;

        elements.propertyModal.classList.add('active');
        elements.propertyModal.setAttribute('aria-hidden', 'false');
    } catch (error) {
        showToast(error.message, 'error');
    }
}

function cacheProperties(properties) {
    properties.forEach((property) => {
        const lookupKey = getPropertyLookupKey(property);
        state.propertyIndex[lookupKey] = property;
    });
}

function getPropertyLookupKey(property) {
    return String(property.lookup_key || property.id);
}

function getCachedProperty(propertyRef) {
    return state.propertyIndex[String(propertyRef)] || null;
}

function findLocalPropertyId(propertyRef) {
    const value = String(propertyRef || '');
    if (/^local:\d+$/.test(value)) {
        return Number(value.split(':')[1]);
    }
    if (/^\d+$/.test(value)) {
        return Number(value);
    }
    return null;
}

async function addToFavorites(propertyId) {
    if (!state.isLoggedIn || !state.currentUser) {
        setFeedback('Log in first to save a property to favorites.', 'info');
        openAuthModal('login');
        return;
    }

    try {
        await fetchAPI(`/api/users/${state.currentUser.id}/favorites`, {
            method: 'POST',
            body: { property_id: propertyId }
        });
        setFeedback('Property saved to favorites. Your recommendation feed just got smarter.', 'success');
        showToast('Added to favorites.', 'success');
        await loadRecommendations();
    } catch (error) {
        setFeedback(error.message, 'error');
    }
}

function logout() {
    state.isLoggedIn = false;
    state.currentUser = null;
    state.aiConversations = [];
    state.activeConversationId = null;
    persistAuth();
    updateAuthUI();
    setFeedback('You have been logged out.', 'info');
    loadRecommendations();
    if (state.activeAiAgentId) {
        const agent = state.aiAgents.find((item) => item.id === state.activeAiAgentId);
        if (agent) {
            resetGuestAiConversation(agent);
        }
    }
}

function updatePagination(current, total) {
    if (!total || total <= 1) {
        elements.propertiesPagination.innerHTML = '';
        return;
    }

    const buttons = [];
    for (let page = 1; page <= total; page += 1) {
        buttons.push(`
            <button class="pagination-btn ${page === current ? 'active' : ''}" data-page="${page}" type="button">
                ${page}
            </button>
        `);
    }

    elements.propertiesPagination.innerHTML = buttons.join('');
}

function setGridState(container, title, description, variant = 'empty') {
    container.innerHTML = `
        <div class="state-card state-${variant}">
            <h3>${escapeHTML(title)}</h3>
            <p>${escapeHTML(description)}</p>
        </div>
    `;
}

function setFeedback(message, type = 'info') {
    if (!message) {
        elements.feedback.className = 'feedback-banner hidden';
        elements.feedback.textContent = '';
        return;
    }

    elements.feedback.textContent = message;
    elements.feedback.className = `feedback-banner ${type}`;
}

function updateRecommendationSummary(message) {
    elements.recommendationSummary.textContent = message;
}

function sanitizeFilters(filters) {
    const cleaned = {};

    Object.entries(filters).forEach(([key, value]) => {
        if (value === undefined || value === null) {
            return;
        }

        const stringValue = String(value).trim();
        if (!stringValue) {
            return;
        }

        cleaned[key] = stringValue;
    });

    return cleaned;
}

function syncFilterInputs(filters) {
    document.getElementById('filter-location').value = filters.location || '';
    document.getElementById('filter-min-price').value = filters.min_price || '';
    document.getElementById('filter-max-price').value = filters.max_price || '';
    document.getElementById('filter-min-bedrooms').value = filters.min_bedrooms || '';
    document.getElementById('filter-property-type').value = filters.property_type || '';
    document.getElementById('filter-status').value = filters.status || '';
}

function buildSearchSummary(filters, count) {
    const chips = [];

    if (filters.location) {
        chips.push(`location "${filters.location}"`);
    }
    if (filters.property_type) {
        chips.push(`${capitalizeLabel(filters.property_type)} homes`);
    }
    if (filters.min_bedrooms) {
        chips.push(`${filters.min_bedrooms}+ bedrooms`);
    }
    if (filters.min_price || filters.max_price) {
        const min = filters.min_price ? `$${formatPrice(filters.min_price)}` : '$0';
        const max = filters.max_price ? `$${formatPrice(filters.max_price)}` : 'open-ended';
        chips.push(`budget ${min} to ${max}`);
    }

    if (!chips.length) {
        return `AI search returned ${count} ${count === 1 ? 'result' : 'results'}.`;
    }

    return `Showing ${count} AI-ranked ${count === 1 ? 'match' : 'matches'} for ${chips.join(', ')}.`;
}

function buildAiNotice(payload) {
    const notes = [];

    if (payload.source === 'local_fallback') {
        notes.push(formatAiFallbackReason(payload.fallback_reason));
    }

    if (payload.listings_source === 'local_inventory') {
        const retrievalNotes = Array.isArray(payload.retrieval_notes) ? payload.retrieval_notes : [];
        if (retrievalNotes.some((note) => String(note).toLowerCase().includes('no explicit location'))) {
            notes.push('No city or neighborhood was given, so listing picks came from the local inventory.');
        }
    } else if (payload.listings_source === 'live_market') {
        notes.push('Listing recommendations came from the live market feed for this request.');
    }

    return notes.filter(Boolean).join(' ');
}

function formatAiFallbackReason(reason) {
    const text = String(reason || '');
    const lowered = text.toLowerCase();

    if (!text) {
        return 'The remote AI model was unavailable, so the assistant used local reasoning for this reply.';
    }

    if (lowered.includes('inference providers') || lowered.includes('403')) {
        return 'The Hugging Face token was rejected for remote inference, so the assistant used local reasoning for this reply.';
    }

    if (lowered.includes('winerror 10013') || lowered.includes('socket') || lowered.includes('forbidden by its access permissions')) {
        return 'The remote AI model could not be reached from this environment, so the assistant used local reasoning for this reply.';
    }

    if (lowered.includes('no completion choices') || lowered.includes('empty message')) {
        return 'The remote AI model returned an unusable response, so the assistant used local reasoning for this reply.';
    }

    return 'The remote AI model could not answer this request, so the assistant used local reasoning for this reply.';
}

function formatConversationTime(value) {
    if (!value) {
        return 'Just now';
    }

    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
        return 'Recently updated';
    }

    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit'
    });
}

function showToast(message, type = 'info') {
    let stack = document.querySelector('.toast-stack');
    if (!stack) {
        stack = document.createElement('div');
        stack.className = 'toast-stack';
        document.body.appendChild(stack);
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    stack.appendChild(toast);

    window.setTimeout(() => {
        toast.remove();
        if (!stack.childElementCount) {
            stack.remove();
        }
    }, 3200);
}

function formatPrice(value) {
    const amount = Number(value || 0);
    return amount.toLocaleString('en-US', {
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    });
}

function formatArea(value) {
    return Number(value || 0).toLocaleString('en-US');
}

function formatCompactNumber(value) {
    return Number(value || 0).toLocaleString('en-US', {
        notation: Number(value || 0) >= 1000 ? 'compact' : 'standard',
        maximumFractionDigits: 1
    });
}

function capitalizeLabel(value) {
    return String(value)
        .split('_')
        .join(' ')
        .replace(/\b\w/g, (character) => character.toUpperCase());
}

function escapeHTML(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function getInitials(name) {
    return String(name || 'RA')
        .split(' ')
        .filter(Boolean)
        .slice(0, 2)
        .map((part) => part[0].toUpperCase())
        .join('');
}

function getCoverStyle(value) {
    return COVER_STYLES.has(value) ? value : 'sunset-grove';
}

window.showPropertyDetail = showPropertyDetail;
