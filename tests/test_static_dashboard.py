from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_stitch_dashboard_keeps_app_js_mount_points():
    html = (ROOT / "static" / "index.html").read_text()

    assert "RhythmFlow" in html
    assert "stitch-dashboard" in html

    required_ids = [
        "chat-history-list",
        "clear-chat-btn",
        "export-history-btn",
        "session-id-display",
        "chat-container",
        "recommendations-container",
        "query-input",
        "send-btn",
        "generate-playlist-btn",
        "loading-spinner",
        "genres-tags",
        "moods-tags",
        "energy-slider",
        "energy-display",
        "total-feedback",
        "approval-rate",
        "thumbs-up",
        "thumbs-down",
        "user-id-input",
        "save-profile-btn",
        "load-profile-btn",
        "profile-select",
        "export-feedback-btn",
        "dark-mode-toggle",
        "playlist-modal",
        "playlist-content",
    ]

    for element_id in required_ids:
        assert f'id="{element_id}"' in html

    assert '<script src="app.js"></script>' in html


def test_app_js_uses_existing_api_and_stitch_card_classes():
    js = (ROOT / "static" / "app.js").read_text()

    assert "const API_BASE = 'http://localhost:5001/api';" in js
    assert "makeAPIRequest('/chat', 'POST'" in js
    assert "recommendation-card" in js
    assert "material-symbols-outlined" in js
    assert "submitFeedback" in js


def test_app_js_sanitizes_raw_gemini_errors_before_display():
    js = (ROOT / "static" / "app.js").read_text()

    assert "sanitizeExplanation" in js
    assert "AI explanation is temporarily unavailable because the model is busy." in js
    assert "showMessage(sanitizeExplanation(response.llm_explanation));" in js
