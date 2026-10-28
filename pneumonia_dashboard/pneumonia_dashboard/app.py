# ══════════════════════════════════════════════════════════════════════
# app.py — Pneumonia Detection Dashboard (VS Code / Local)
# Run:  python app.py
# ══════════════════════════════════════════════════════════════════════

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TF C++ logs

# GPU memory growth — must be before any TF ops
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

import gradio as gr
from datetime import datetime
from model_loader import loader
from config import HOST, PORT, SHARE

# ── Load model on startup ──────────────────────────────────────────────
print('\n' + '═'*55)
print('  🫁  Pneumonia Detection Dashboard')
print('═'*55)
loader.load()
print(f'  Status    : {loader.status.upper()}')
print(f'  Model     : {loader.model_name}')
print(f'  Threshold : {loader.threshold:.4f}')
print(f'  Img size  : {loader.img_size}×{loader.img_size}')
print('═'*55 + '\n')

# ── Session state ──────────────────────────────────────────────────────
log_entries  = []
session_stats = {'total': 0, 'pneumonia': 0, 'normal': 0}


# ── HTML builders ──────────────────────────────────────────────────────

def build_result_card(r):
    return f"""
    <div style="font-family:'Courier New',monospace;
                background:#0a0a14;
                border:2px solid {r['label_color']};
                border-radius:10px;
                padding:20px 24px;">
      <div style="font-size:2.2em; font-weight:900;
                  color:{r['label_color']}; letter-spacing:3px;">
        {r['icon']}&nbsp; {r['label']}
      </div>
      <div style="margin:10px 0 14px; color:#9ca3af; font-size:0.9em; line-height:1.8;">
        P(Pneumonia)&nbsp;:&nbsp;
          <b style="color:{r['label_color']}">{r['prob_pct']:.1f}%</b>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Confidence&nbsp;:&nbsp;
          <b style="color:{r['label_color']}">{r['conf_pct']:.1f}%</b>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Threshold&nbsp;:&nbsp;{r['threshold']:.4f}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        Speed&nbsp;:&nbsp;{r['elapsed_ms']:.0f}ms
      </div>
      <span style="background:{r['cert_color']}22;
                   border:1px solid {r['cert_color']};
                   color:{r['cert_color']};
                   border-radius:5px;
                   padding:4px 12px;
                   font-size:0.78em;
                   letter-spacing:1px;">
        {r['certainty']}
      </span>
      {'<span style="margin-left:10px;background:#f9731622;border:1px solid #f97316;color:#f97316;border-radius:5px;padding:4px 10px;font-size:0.75em">DEMO MODE</span>' if r['demo'] else ''}
    </div>
    """


def build_log_html():
    if not log_entries:
        return '<p style="color:#374151;font-family:monospace;padding:8px">No predictions yet</p>'

    rows = ''
    for i, e in enumerate(log_entries):
        bg = '#0a0a14' if i % 2 == 0 else '#0d0d1a'
        rows += f"""
        <tr style="background:{bg}">
          <td style="padding:7px 12px;color:#4b5563">{e['time']}</td>
          <td style="padding:7px 12px;color:{e['color']};font-weight:700">{e['icon']} {e['label']}</td>
          <td style="padding:7px 12px;color:#9ca3af">{e['prob']}</td>
          <td style="padding:7px 12px;color:#9ca3af">{e['conf']}</td>
          <td style="padding:7px 12px;color:#4b5563">{e['speed']}</td>
          <td style="padding:7px 12px">
            <span style="background:{e['cert_bg']};border:1px solid {e['cert_color']};
                         color:{e['cert_color']};border-radius:4px;padding:2px 8px;
                         font-size:0.7em">{e['certainty']}</span>
          </td>
        </tr>"""

    return f"""
    <table style="width:100%;border-collapse:collapse;
                  font-family:'Courier New',monospace;font-size:0.82em;">
      <thead>
        <tr style="background:#111827;color:#374151;
                   text-transform:uppercase;letter-spacing:1px;font-size:0.72em;">
          <th style="padding:8px 12px;text-align:left">Time</th>
          <th style="padding:8px 12px;text-align:left">Result</th>
          <th style="padding:8px 12px;text-align:left">P(Pneumonia)</th>
          <th style="padding:8px 12px;text-align:left">Confidence</th>
          <th style="padding:8px 12px;text-align:left">Speed</th>
          <th style="padding:8px 12px;text-align:left">Certainty</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def build_stats_html():
    t = session_stats['total']
    if t == 0:
        return '<p style="color:#374151;font-family:monospace;padding:8px">No predictions yet</p>'

    p_pct = session_stats['pneumonia'] / t * 100
    n_pct = session_stats['normal']    / t * 100

    return f"""
    <div style="display:flex;gap:16px;flex-wrap:wrap;font-family:'Courier New',monospace;">
      <div style="background:#0a0a14;border:1px solid #1f2937;
                  border-radius:8px;padding:12px 22px;text-align:center;min-width:90px;">
        <div style="font-size:1.9em;font-weight:900;color:#f9fafb">{t}</div>
        <div style="font-size:0.68em;color:#4b5563;letter-spacing:1px;margin-top:2px">TOTAL</div>
      </div>
      <div style="background:#0a0a14;border:1px solid #ef444433;
                  border-radius:8px;padding:12px 22px;text-align:center;min-width:90px;">
        <div style="font-size:1.9em;font-weight:900;color:#ef4444">{session_stats['pneumonia']}</div>
        <div style="font-size:0.68em;color:#4b5563;letter-spacing:1px;margin-top:2px">
          PNEUMONIA · {p_pct:.0f}%
        </div>
      </div>
      <div style="background:#0a0a14;border:1px solid #22c55e33;
                  border-radius:8px;padding:12px 22px;text-align:center;min-width:90px;">
        <div style="font-size:1.9em;font-weight:900;color:#22c55e">{session_stats['normal']}</div>
        <div style="font-size:0.68em;color:#4b5563;letter-spacing:1px;margin-top:2px">
          NORMAL · {n_pct:.0f}%
        </div>
      </div>
    </div>"""


def build_model_info():
    status_color = {'loaded': '#22c55e', 'demo': '#f97316', 'not_loaded': '#ef4444'}
    sc = status_color.get(loader.status, '#6b7280')
    return f"""
    <div style="font-family:'Courier New',monospace;font-size:0.75em;
                color:#374151;border-top:1px solid #111827;padding-top:10px;line-height:1.9;">
      <span style="color:{sc}">⬤</span>
      &nbsp;{loader.status.upper()}
      &nbsp;&nbsp;|&nbsp;&nbsp;{loader.model_name}
      &nbsp;&nbsp;|&nbsp;&nbsp;threshold {loader.threshold:.4f}
      &nbsp;&nbsp;|&nbsp;&nbsp;{loader.img_size}×{loader.img_size}px
      &nbsp;&nbsp;|&nbsp;&nbsp;TF {tf.__version__}
    </div>"""


# ── Core predict function ──────────────────────────────────────────────

IDLE_CARD = '<p style="color:#374151;font-family:monospace;padding:8px">Upload an X-ray to begin</p>'

def run_predict(pil_img):
    if pil_img is None:
        return IDLE_CARD, 0, build_log_html(), build_stats_html()

    r = loader.predict(pil_img)

    # Update log
    cert_colors = {
        'HIGH CONFIDENCE'        : ('#4ade80', '#4ade8022'),
        'MODERATE CONFIDENCE'    : ('#facc15', '#facc1522'),
        'LOW — REVIEW RECOMMENDED': ('#f97316', '#f9731622'),
    }
    cc, cb = cert_colors.get(r['certainty'], ('#6b7280', '#6b728022'))

    log_entries.insert(0, {
        'time'      : datetime.now().strftime('%H:%M:%S'),
        'label'     : r['label'],
        'icon'      : r['icon'],
        'prob'      : f"{r['prob_pct']:.1f}%",
        'conf'      : f"{r['conf_pct']:.1f}%",
        'speed'     : f"{r['elapsed_ms']:.0f}ms",
        'color'     : r['label_color'],
        'certainty' : r['certainty'],
        'cert_color': cc,
        'cert_bg'   : cb,
    })
    if len(log_entries) > 15:
        log_entries.pop()

    # Update session stats
    session_stats['total'] += 1
    key = 'pneumonia' if r['label'] == 'PNEUMONIA' else 'normal'
    session_stats[key] += 1

    return (
        build_result_card(r),
        round(r['conf_pct'], 1),
        build_log_html(),
        build_stats_html(),
    )


def clear_all():
    log_entries.clear()
    session_stats.update({'total': 0, 'pneumonia': 0, 'normal': 0})
    return (
        None,
        IDLE_CARD,
        0,
        '<p style="color:#374151;font-family:monospace;padding:8px">No predictions yet</p>',
        '<p style="color:#374151;font-family:monospace;padding:8px">No predictions yet</p>',
    )


# ── CSS ────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; }

body, .gradio-container {
    background: #060610 !important;
    font-family: 'Courier New', monospace !important;
}

.gr-panel, .gr-box, .gr-form, .gr-block, .block {
    background: #0a0a14 !important;
    border-color: #1f2937 !important;
}

/* Labels & headings */
label, .gr-label, span.svelte-1gfkn6j, p {
    color: #9ca3af !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.8em !important;
    letter-spacing: 0.5px !important;
}

/* Buttons */
.gr-button {
    font-family: 'Courier New', monospace !important;
    letter-spacing: 1px !important;
    font-weight: 700 !important;
    border-radius: 6px !important;
}
.gr-button-primary {
    background: #2563eb !important;
    border: none !important;
    color: #fff !important;
}
.gr-button-primary:hover { background: #1d4ed8 !important; }

.gr-button-secondary {
    background: #0a0a14 !important;
    border: 1px solid #1f2937 !important;
    color: #6b7280 !important;
}
.gr-button-secondary:hover { border-color: #374151 !important; color:#9ca3af !important; }

/* Slider */
input[type=range] { accent-color: #2563eb !important; }

/* Image upload */
.image-container, .upload-btn-wrapper {
    background: #0a0a14 !important;
    border: 1px dashed #1f2937 !important;
    border-radius: 8px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #060610; }
::-webkit-scrollbar-thumb { background: #1f2937; border-radius: 2px; }
"""


# ── Gradio UI ──────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, title='🫁 Pneumonia Dashboard') as app:

    # Header
    gr.HTML(f"""
    <div style="padding:24px 0 16px;border-bottom:1px solid #111827;margin-bottom:20px;">
      <div style="font-family:'Courier New',monospace;">
        <div style="font-size:1.8em;font-weight:900;letter-spacing:4px;color:#f9fafb;">
          🫁 &nbsp;PNEUMONIA DETECTION
        </div>
        <div style="color:#374151;font-size:0.72em;letter-spacing:2px;margin-top:4px;">
          CHEST X-RAY ANALYSIS &nbsp;·&nbsp; REAL-TIME INFERENCE &nbsp;·&nbsp;
          {'<span style="color:#22c55e">MODEL LOADED</span>' if loader.status=='loaded'
           else '<span style="color:#f97316">DEMO MODE</span>'}
        </div>
      </div>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── Left: Upload ───────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            img_in = gr.Image(
                type='pil',
                label='CHEST X-RAY',
                height=310,
                show_label=True,
            )
            gr.HTML('<div style="height:8px"></div>')
            with gr.Row():
                run_btn   = gr.Button('▶ ANALYZE',  variant='primary',   scale=2)
                clear_btn = gr.Button('✕ CLEAR',    variant='secondary', scale=1)

            gr.HTML(build_model_info())

        # ── Right: Result ──────────────────────────────────────────────
        with gr.Column(scale=1, min_width=320):
            result_html = gr.HTML(value=IDLE_CARD, label='DIAGNOSIS')
            gr.HTML('<div style="height:6px"></div>')
            conf_bar = gr.Slider(
                minimum=0, maximum=100, value=0,
                label='CONFIDENCE %',
                interactive=False,
            )

    # Stats row
    gr.HTML('<div style="border-top:1px solid #0d0d1a;margin:16px 0 8px;"></div>')
    gr.HTML('<div style="font-family:monospace;color:#1f2937;font-size:0.7em;letter-spacing:2px;margin-bottom:8px">SESSION STATISTICS</div>')
    stats_html = gr.HTML(value='<p style="color:#374151;font-family:monospace;padding:8px">No predictions yet</p>')

    # Log
    gr.HTML('<div style="border-top:1px solid #0d0d1a;margin:16px 0 8px;"></div>')
    gr.HTML('<div style="font-family:monospace;color:#1f2937;font-size:0.7em;letter-spacing:2px;margin-bottom:8px">PREDICTION LOG</div>')
    log_html = gr.HTML(value='<p style="color:#374151;font-family:monospace;padding:8px">No predictions yet</p>')

    # ── Wire up ────────────────────────────────────────────────────────
    outputs = [result_html, conf_bar, log_html, stats_html]

    run_btn.click(fn=run_predict, inputs=[img_in], outputs=outputs)
    img_in.change(fn=run_predict, inputs=[img_in], outputs=outputs)   # auto on upload
    clear_btn.click(
        fn=clear_all,
        outputs=[img_in, result_html, conf_bar, log_html, stats_html]
    )


# ── Launch ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'Dashboard running at: http://{HOST}:{PORT}')
    print('Press Ctrl+C to stop.\n')
    app.launch(server_name=HOST, server_port=PORT, share=SHARE, inbrowser=True)
