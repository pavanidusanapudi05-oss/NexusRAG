def get_custom_css() -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* Top Header Banner */
    .nexus-hero {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #020617 100%);
        border: 1px solid #312e81;
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .nexus-hero h1 {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .nexus-hero p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    /* Enterprise Glassmorphic Cards */
    .nexus-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 18px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .nexus-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
    }

    /* Evidence Card Styling */
    .evidence-card {
        background: #0f172a;
        border-left: 4px solid #6366f1;
        border-top: 1px solid #1e293b;
        border-right: 1px solid #1e293b;
        border-bottom: 1px solid #1e293b;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .evidence-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .evidence-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #f1f5f9;
    }

    .evidence-meta {
        font-size: 0.8rem;
        color: #94a3b8;
    }

    .evidence-snippet {
        font-size: 0.9rem;
        color: #cbd5e1;
        background: #1e293b;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        line-height: 1.5;
        border-left: 2px solid #3b82f6;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .badge-high {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-medium {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .badge-low {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .badge-intent {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .badge-diff-added {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
    }

    .badge-diff-removed {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
    }

    .badge-diff-modified {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
    }

    .badge-diff-unchanged {
        background: rgba(148, 163, 184, 0.2);
        color: #94a3b8;
        border: 1px solid #64748b;
    }

    /* Metric Box */
    .metric-box {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 2px;
    }

    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """
