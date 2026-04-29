/**
 * RAG Music Recommendation System - Frontend JavaScript
 * 
 * Handles all client-side interactions with the Flask backend API:
 * - Chat interface with message display
 * - API communication for recommendations
 * - User feedback logging
 * - Profile management
 * - Playlist generation
 */

// ============================================================================
// Global State
// ============================================================================

let SESSION_ID = localStorage.getItem('session_id') || null;
const API_BASE = 'http://localhost:5001/api';

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing RAG Music Recommendation System...');
    
    // Create or restore session
    if (!SESSION_ID) {
        SESSION_ID = generateUUID();
        localStorage.setItem('session_id', SESSION_ID);
    }
    
    updateSessionDisplay();
    attachEventListeners();
    loadAvailableProfiles();
    restoreTheme();
});

// ============================================================================
// Helper Functions
// ============================================================================

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

function updateSessionDisplay() {
    const display = document.getElementById('session-id-display');
    display.textContent = `Session: ${SESSION_ID.substring(0, 8)}...`;
}

function escapeHTML(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }[char]));
}

function sanitizeExplanation(explanation) {
    const message = String(explanation ?? '');
    if (
        message.includes('[Gemini Error:') &&
        (message.includes('503') || message.includes('UNAVAILABLE') || message.toLowerCase().includes('high demand'))
    ) {
        return 'AI explanation is temporarily unavailable because the model is busy. Showing recommendations based on similarity search.';
    }
    return message;
}

function showLoading() {
    document.getElementById('loading-spinner').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-spinner').style.display = 'none';
}

function showMessage(text, isUser = false) {
    const container = document.getElementById('chat-container');
    
    // Remove welcome message if present
    const welcome = container.querySelector('.welcome-message');
    if (welcome) welcome.remove();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = isUser
        ? 'message user flex items-start gap-sm max-w-[85%] self-end flex-row-reverse'
        : 'message assistant flex items-start gap-sm max-w-[85%]';

    const avatar = document.createElement('div');
    avatar.className = isUser
        ? 'w-8 h-8 rounded-full bg-secondary-container text-on-secondary-container flex items-center justify-center shrink-0'
        : 'w-8 h-8 rounded-full bg-primary-container text-primary flex items-center justify-center shrink-0';
    avatar.innerHTML = `<span class="material-symbols-outlined text-[18px]">${isUser ? 'person' : 'graphic_eq'}</span>`;

    const bubbleWrapper = document.createElement('div');
    bubbleWrapper.className = 'flex flex-col gap-1 max-w-full';

    const bubble = document.createElement('div');
    bubble.className = isUser
        ? 'message-bubble bg-primary text-on-primary p-sm rounded-l-xl rounded-tr-xl rounded-br-sm shadow-sm'
        : 'message-bubble bg-surface-variant text-on-surface-variant p-sm rounded-r-xl rounded-tl-xl rounded-bl-sm shadow-sm';
    bubble.textContent = text;

    const time = document.createElement('div');
    time.className = isUser
        ? 'message-time text-[11px] text-outline text-right'
        : 'message-time text-[11px] text-outline';
    time.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    bubbleWrapper.appendChild(bubble);
    bubbleWrapper.appendChild(time);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(bubbleWrapper);
    const recommendationsContainer = document.getElementById('recommendations-container');
    if (recommendationsContainer && recommendationsContainer.parentElement === container) {
        container.insertBefore(messageDiv, recommendationsContainer);
    } else {
        container.appendChild(messageDiv);
    }
    
    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
}

function scrollToBottom() {
    const container = document.getElementById('chat-container');
    container.scrollTop = container.scrollHeight;
}

// ============================================================================
// API Communication
// ============================================================================

async function makeAPIRequest(endpoint, method = 'POST', body = null) {
    try {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
        };
        
        if (body) {
            options.body = JSON.stringify(body);
        }
        
        const response = await fetch(`${API_BASE}${endpoint}`, options);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || `API Error: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        throw error;
    }
}

// ============================================================================
// Chat & Recommendations
// ============================================================================

async function sendQuery() {
    const input = document.getElementById('query-input');
    const query = input.value.trim();
    
    if (!query) return;
    
    // Show user message
    showMessage(query, true);
    addToChatHistory(query);
    
    // Clear input
    input.value = '';
    
    // Show loading
    showLoading();
    
    try {
        // Get recommendations from API
        const response = await makeAPIRequest('/chat', 'POST', {
            query,
            session_id: SESSION_ID,
            k: 5
        });
        
        // Update session ID if new
        if (response.session_id) {
            SESSION_ID = response.session_id;
            localStorage.setItem('session_id', SESSION_ID);
            updateSessionDisplay();
        }
        
        // Show AI explanation
        showMessage(sanitizeExplanation(response.llm_explanation));
        
        // Display recommendation cards
        displayRecommendations(response.recommendations);
        
        // Update user profile
        updateProfileDisplay(response.user_profile);
        
    } catch (error) {
        showMessage(`Error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendations-container');
    container.innerHTML = '';
    
    if (!recommendations || recommendations.length === 0) {
        container.innerHTML = '<p class="empty-state">No recommendations available</p>';
        return;
    }
    
    const grid = document.createElement('div');
    grid.className = 'recommendations-grid';
    
    recommendations.forEach((rec, index) => {
        const card = createRecommendationCard(rec, index);
        grid.appendChild(card);
    });
    
    container.appendChild(grid);
}

function createRecommendationCard(recommendation, index) {
    const song = recommendation.song;
    const score = recommendation.score;
    
    const card = document.createElement('div');
    card.className = 'recommendation-card bg-surface rounded-lg border border-outline-variant p-sm flex items-center gap-md hover:border-primary transition-colors group cursor-pointer shadow-sm hover:shadow-md relative overflow-hidden';
    card.dataset.songId = song.id;
    
    const energyLevel = ['Low', 'Low-Mid', 'Mid', 'Mid-High', 'High'];
    const energyIndex = Math.min(Math.floor(song.energy * 5), energyLevel.length - 1);
    const batteryIcon = song.energy < 0.35 ? 'battery_2_bar' : song.energy < 0.7 ? 'battery_3_bar' : 'battery_5_bar';
    
    card.innerHTML = `
        <div class="absolute inset-0 bg-gradient-to-r from-transparent to-surface-variant opacity-0 group-hover:opacity-20 transition-opacity"></div>

        <div class="w-16 h-16 rounded-md overflow-hidden shrink-0 relative bg-surface-container flex items-center justify-center">
            <span class="material-symbols-outlined text-primary text-[32px]">album</span>
            <div class="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-[2px]">
                <span class="material-symbols-outlined text-white icon-fill">play_arrow</span>
            </div>
        </div>
        
        <div class="flex-1 flex flex-col justify-center min-w-0 relative">
            <div class="flex items-center gap-xs min-w-0">
                <h4 class="font-title-sm text-title-sm text-on-surface truncate">${escapeHTML(song.title)}</h4>
                <span class="bg-primary-container text-on-primary-container px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider shrink-0">${(score * 100).toFixed(0)}% Match</span>
            </div>
            <p class="font-body-md text-body-md text-on-surface-variant text-sm truncate">${escapeHTML(song.artist)}</p>
            <p class="font-body-md text-[13px] text-outline line-clamp-2 mt-1">${escapeHTML(song.description || 'A great track for your taste.')}</p>
            <div class="card-badges flex items-center gap-xs mt-2">
                <span class="badge genre font-label-sm text-[11px] text-tertiary bg-surface-container px-2 py-0.5 rounded-sm">${escapeHTML(song.genre)}</span>
                <span class="badge mood font-label-sm text-[11px] text-tertiary bg-surface-container px-2 py-0.5 rounded-sm">${escapeHTML(song.mood)}</span>
            </div>
        </div>
        
        <div class="flex flex-col items-end gap-xs shrink-0 pl-sm border-l border-outline-variant/50 relative">
            <div class="flex items-center gap-1 text-xs text-on-surface-variant" title="Energy Level: ${energyLevel[energyIndex]}">
                <span class="material-symbols-outlined text-[16px]">${batteryIcon}</span>
                <span>${energyLevel[energyIndex]}</span>
            </div>
            <div class="text-xs text-outline">${escapeHTML(song.tempo_bpm)} BPM</div>
            <div class="card-actions flex items-center gap-xs mt-1">
                <button class="feedback-btn thumbs-up p-1 rounded hover:bg-surface-variant text-outline hover:text-primary transition-colors" title="Like this song">
                    <span class="material-symbols-outlined text-[18px]">thumb_up</span>
                </button>
                <button class="feedback-btn thumbs-down p-1 rounded hover:bg-surface-variant text-outline hover:text-error transition-colors" title="Dislike this song">
                    <span class="material-symbols-outlined text-[18px]">thumb_down</span>
                </button>
            </div>
        </div>
    `;

    card.querySelector('.thumbs-up').addEventListener('click', (event) => {
        event.stopPropagation();
        submitFeedback(song.id, 1, song.title);
    });
    card.querySelector('.thumbs-down').addEventListener('click', (event) => {
        event.stopPropagation();
        submitFeedback(song.id, -1, song.title);
    });
    
    return card;
}

function updateProfileDisplay(userProfile) {
    // Update genres
    const genresTags = document.getElementById('genres-tags');
    if (userProfile.favorite_genres && userProfile.favorite_genres.length > 0) {
        genresTags.innerHTML = userProfile.favorite_genres
            .map(g => `<span class="tag bg-secondary-container text-on-secondary-container rounded-full px-3 py-1 font-label-sm text-[12px] border border-secondary/20">${escapeHTML(g)}</span>`)
            .join('');
    } else {
        genresTags.innerHTML = '<span class="tag bg-secondary-container text-on-secondary-container rounded-full px-3 py-1 font-label-sm text-[12px] border border-secondary/20">None yet</span>';
    }
    
    // Update moods
    const moodsTags = document.getElementById('moods-tags');
    if (userProfile.favorite_moods && userProfile.favorite_moods.length > 0) {
        moodsTags.innerHTML = userProfile.favorite_moods
            .map(m => `<span class="tag bg-tertiary-container/30 text-on-surface rounded-full px-3 py-1 font-label-sm text-[12px] border border-tertiary/20">${escapeHTML(m)}</span>`)
            .join('');
    } else {
        moodsTags.innerHTML = '<span class="tag bg-tertiary-container/30 text-on-surface rounded-full px-3 py-1 font-label-sm text-[12px] border border-tertiary/20">None yet</span>';
    }
    
    // Update energy slider
    const energySlider = document.getElementById('energy-slider');
    const energyDisplay = document.getElementById('energy-display');
    if (userProfile.target_energy !== undefined) {
        energySlider.value = userProfile.target_energy;
        energyDisplay.textContent = (userProfile.target_energy * 100).toFixed(0) + '%';
    }
}

// ============================================================================
// Feedback System
// ============================================================================

async function submitFeedback(songId, rating, songTitle = '') {
    try {
        const response = await makeAPIRequest('/feedback', 'POST', {
            session_id: SESSION_ID,
            song_id: songId,
            song_title: songTitle,
            rating: rating,
            query: ''
        });
        
        // Update feedback stats
        updateFeedbackStats(response.stats);
        
        // Update button appearance
        const card = document.querySelector(`[data-song-id="${songId}"]`);
        if (card) {
            const btn = rating > 0 
                ? card.querySelector('.feedback-btn.thumbs-up')
                : card.querySelector('.feedback-btn.thumbs-down');
            if (btn) {
                btn.classList.add(rating > 0 ? 'liked' : 'disliked');
            }
        }
        
        showMessage(`Feedback recorded! ${rating > 0 ? '👍' : '👎'}`);
        
    } catch (error) {
        showMessage(`Error logging feedback: ${error.message}`);
    }
}

function updateFeedbackStats(stats) {
    document.getElementById('total-feedback').textContent = stats.total_feedback || '0';
    document.getElementById('thumbs-up').textContent = stats.thumbs_up || '0';
    document.getElementById('thumbs-down').textContent = stats.thumbs_down || '0';
    
    const approvalRate = stats.approval_rate !== undefined 
        ? (stats.approval_rate * 100).toFixed(0) + '%'
        : 'N/A';
    document.getElementById('approval-rate').textContent = approvalRate;
}

// ============================================================================
// Playlist Generation
// ============================================================================

async function generatePlaylist() {
    showLoading();
    
    try {
        const response = await makeAPIRequest('/playlist', 'POST', {
            session_id: SESSION_ID,
            duration_minutes: 30,
            k: 10
        });
        
        displayPlaylist(response.playlist);
        
    } catch (error) {
        showMessage(`Error generating playlist: ${error.message}`);
    } finally {
        hideLoading();
    }
}

function displayPlaylist(playlist) {
    const modal = document.getElementById('playlist-modal');
    const content = document.getElementById('playlist-content');
    
    let html = `
        <h3>${playlist.name}</h3>
        <p><strong>Duration:</strong> ${playlist.total_duration_minutes} minutes</p>
        <p><strong>Songs:</strong> ${playlist.total_songs}</p>
        
        <table class="playlist-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Title</th>
                    <th>Artist</th>
                    <th>Genre</th>
                    <th>Energy</th>
                </tr>
            </thead>
            <tbody>
    `;
    
    playlist.songs.forEach(song => {
        const energyPercent = (song.energy * 100).toFixed(0);
        html += `
            <tr>
                <td>${song.rank}</td>
                <td>${song.title}</td>
                <td>${song.artist}</td>
                <td>${song.genre}</td>
                <td>${energyPercent}%</td>
            </tr>
        `;
    });
    
    html += `
            </tbody>
        </table>
    `;
    
    content.innerHTML = html;
    modal.style.display = 'flex';
}

function closePlaylistModal() {
    document.getElementById('playlist-modal').style.display = 'none';
}

// ============================================================================
// Profile Management
// ============================================================================

async function saveProfile() {
    const userId = document.getElementById('user-id-input').value.trim();
    
    if (!userId) {
        alert('Please enter a profile name');
        return;
    }
    
    try {
        const response = await makeAPIRequest('/profile', 'POST', {
            session_id: SESSION_ID,
            action: 'save',
            user_id: userId
        });
        
        showMessage(`Profile "${userId}" saved! ✅`);
        loadAvailableProfiles();
        
    } catch (error) {
        showMessage(`Error saving profile: ${error.message}`);
    }
}

async function loadProfile() {
    const select = document.getElementById('profile-select');
    const userId = select.value;
    
    if (!userId) {
        alert('Please select a profile');
        return;
    }
    
    try {
        const response = await makeAPIRequest(`/profile?session_id=${SESSION_ID}&user_id=${userId}`, 'GET');
        
        if (response.success) {
            updateProfileDisplay(response.profile);
            showMessage(`Profile "${userId}" loaded! 📂`);
        }
        
    } catch (error) {
        showMessage(`Error loading profile: ${error.message}`);
    }
}

async function loadAvailableProfiles() {
    try {
        const response = await makeAPIRequest('/profiles', 'GET');
        const select = document.getElementById('profile-select');
        
        // Clear existing options except first one
        while (select.options.length > 1) {
            select.remove(1);
        }
        
        // Add available profiles
        if (response.profiles && response.profiles.length > 0) {
            response.profiles.forEach(profile => {
                const option = document.createElement('option');
                option.value = profile;
                option.textContent = profile;
                select.appendChild(option);
            });
        }
        
    } catch (error) {
        console.error('Error loading profiles:', error);
    }
}

async function exportFeedback() {
    const userId = document.getElementById('user-id-input').value.trim();
    
    if (!userId) {
        alert('Please enter a profile name first');
        return;
    }
    
    try {
        // Create a download link for the CSV
        const link = document.createElement('a');
        link.href = `${API_BASE}/export/${userId}`;
        link.download = `feedback_${userId}.csv`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        showMessage(`Feedback exported! 📥`);
        
    } catch (error) {
        showMessage(`Error exporting feedback: ${error.message}`);
    }
}

// ============================================================================
// Chat History
// ============================================================================

function addToChatHistory(query) {
    const historyList = document.getElementById('chat-history-list');
    
    // Remove empty state if present
    const empty = historyList.querySelector('.empty-state');
    if (empty) empty.remove();
    
    const item = document.createElement('button');
    item.className = 'chat-history-item flex flex-col items-start p-xs rounded-lg hover:bg-slate-100 dark:hover:bg-slate-900 transition-colors text-left w-full group';
    item.innerHTML = `
        <span class="text-on-surface-variant truncate w-full text-sm font-medium group-hover:text-primary transition-colors">${escapeHTML(query.substring(0, 50) + (query.length > 50 ? '...' : ''))}</span>
        <span class="text-xs text-slate-400">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
    `;
    item.title = query;
    item.onclick = () => {
        document.getElementById('query-input').value = query;
    };
    
    historyList.insertBefore(item, historyList.firstChild);
}

function clearChatHistory() {
    if (confirm('Are you sure you want to clear chat history?')) {
        document.getElementById('chat-history-list').innerHTML = '<p class="empty-state text-sm text-slate-400 px-2 py-4">No chat history yet</p>';
        document.getElementById('chat-container').innerHTML = `
            <div class="welcome-message flex items-start gap-sm max-w-[85%]">
                <div class="w-8 h-8 rounded-full bg-primary-container text-primary flex items-center justify-center shrink-0">
                    <span class="material-symbols-outlined text-[18px]">graphic_eq</span>
                </div>
                <div class="bg-surface-variant text-on-surface-variant p-sm rounded-r-xl rounded-tl-xl rounded-bl-sm shadow-sm">
                    <p class="font-body-md text-body-md">Welcome back. Ask for music recommendations in natural language, and I will tune them with your RAG profile.</p>
                </div>
            </div>
            <div id="recommendations-container" class="recommendations-container flex flex-col gap-sm w-full mt-xs"></div>
        `;
    }
}

function exportHistory() {
    const historyList = document.getElementById('chat-history-list');
    const items = Array.from(historyList.querySelectorAll('.chat-history-item'));
    
    if (items.length === 0) {
        alert('No chat history to export');
        return;
    }
    
    const history = items.map(item => item.title || item.textContent.trim()).join('\n');
    const blob = new Blob([history], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `chat_history_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    URL.revokeObjectURL(url);
}

// ============================================================================
// Theme Management
// ============================================================================

function toggleDarkMode() {
    const isDark = document.documentElement.classList.toggle('dark');
    document.documentElement.classList.toggle('light', !isDark);
    document.body.classList.toggle('dark-mode', isDark);
    localStorage.setItem('dark-mode', isDark);
}

function restoreTheme() {
    const isDark = localStorage.getItem('dark-mode') === 'true';
    document.documentElement.classList.toggle('dark', isDark);
    document.documentElement.classList.toggle('light', !isDark);
    document.body.classList.toggle('dark-mode', isDark);
}

// ============================================================================
// Event Listeners
// ============================================================================

function attachEventListeners() {
    // Chat input
    const queryInput = document.getElementById('query-input');
    queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendQuery();
        }
    });
    
    // Send button
    document.getElementById('send-btn').addEventListener('click', sendQuery);
    
    // Playlist generation
    document.getElementById('generate-playlist-btn').addEventListener('click', generatePlaylist);
    
    // Profile management
    document.getElementById('save-profile-btn').addEventListener('click', saveProfile);
    document.getElementById('load-profile-btn').addEventListener('click', loadProfile);
    document.getElementById('export-feedback-btn').addEventListener('click', exportFeedback);
    
    // Chat history
    document.getElementById('clear-chat-btn').addEventListener('click', clearChatHistory);
    document.getElementById('export-history-btn').addEventListener('click', exportHistory);
    
    // Dark mode
    document.getElementById('dark-mode-toggle').addEventListener('click', toggleDarkMode);
    
    // Modal close buttons
    document.querySelectorAll('.modal-close').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.target.closest('.modal').style.display = 'none';
        });
    });
    
    // Close modal when clicking outside
    document.querySelectorAll('.modal').forEach(modal => {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
    });
    
    // Energy slider
    document.getElementById('energy-slider').addEventListener('input', (e) => {
        const percent = (parseFloat(e.target.value) * 100).toFixed(0);
        document.getElementById('energy-display').textContent = percent + '%';
    });
}

// ============================================================================
// Initialization
// ============================================================================

console.log('RAG Music Recommendation System loaded successfully');
