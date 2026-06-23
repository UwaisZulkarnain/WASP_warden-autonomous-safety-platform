"""
WASP PATCH — On-Demand Report Generation
=========================================
Apply these 4 changes to wasp_backend.py:

CHANGE 1 — Import report_generator near the top (after existing imports)
CHANGE 2 — Add /api/report/generate Flask endpoint
CHANGE 3 — Add "Generate Report" button HTML to the dashboard
CHANGE 4 — Add JS handler for the button

Each change is shown with the EXACT text to find and what to replace it with.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1
# Find this line in wasp_backend.py:
#     from groq import Groq
# Add the import AFTER it:
# ─────────────────────────────────────────────────────────────────────────────

CHANGE_1_AFTER = "from groq import Groq"
CHANGE_1_INSERT = """
from report_generator import generate_report, save_violation_screenshot
"""

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 2
# Add this Flask endpoint just BEFORE the line:
#     if __name__ == '__main__':
# ─────────────────────────────────────────────────────────────────────────────

CHANGE_2_BEFORE = "if __name__ == '__main__':"
CHANGE_2_INSERT = '''
@app.route('/api/report/generate', methods=['POST'])
def api_generate_report():
    """
    On-demand safety report endpoint.
    Saves a violation screenshot of the current frame,
    then generates a PDF report and returns the file path.
    """
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("reports", exist_ok=True)

        # Grab current frame for screenshot
        with frame_lock:
            frame_copy = latest_frame.copy() if latest_frame is not None else None

        screenshot_path = save_violation_screenshot(frame_copy, cv_state, "reports")

        # Grab latest ML prediction
        with ml_prediction_lock:
            ml_snap = ml_latest_prediction

        output_path = f"reports/safety_report_{ts}.pdf"
        generate_report(
            output_path=output_path,
            sensor_data=sensor_data,
            cv_state=cv_state,
            db_path="wasp.db",
            screenshot_path=screenshot_path,
            ml_result=ml_snap,
        )

        screenshot_name = os.path.basename(screenshot_path) if screenshot_path else None
        report_name = os.path.basename(output_path)

        log_alert("REPORT_GENERATED", f"On-demand report: {report_name}")
        print(f"[REPORT] Generated: {output_path}")

        return jsonify({
            "status": "ok",
            "report": report_name,
            "screenshot": screenshot_name,
            "timestamp": ts,
        })

    except Exception as e:
        print(f"[REPORT Error] {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/reports/<path:filename>')
def serve_report(filename):
    """Serve generated report files for download."""
    from flask import send_from_directory
    return send_from_directory(os.path.abspath("reports"), filename)

'''

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 3  — Button in the HTML header
# Inside HTML_TEMPLATE, find the closing tag of the header-right div:
#             </div>
#         </div>        ← this closing div ends the header
# Replace the header-right block with this updated version that adds the button.
#
# Find this exact string inside HTML_TEMPLATE:
# ─────────────────────────────────────────────────────────────────────────────

CHANGE_3_FIND = '''        <div class="header-right">
            <div class="header-time" id="clock">--:--:--</div>
            <div class="header-date" id="date">--</div>
        </div>'''

CHANGE_3_REPLACE = '''        <div style="display:flex;align-items:center;gap:16px;">
            <button id="report-btn" onclick="generateReport()" style="
                background:#dc2626;border:1.5px solid #fca5a5;color:#fff;
                font-size:12px;font-weight:700;padding:8px 16px;border-radius:7px;
                cursor:pointer;letter-spacing:.5px;transition:all .2s;white-space:nowrap;">
                &#128196; Generate Report
            </button>
            <div class="header-right">
                <div class="header-time" id="clock">--:--:--</div>
                <div class="header-date" id="date">--</div>
            </div>
        </div>'''

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 4  — JS function
# Inside HTML_TEMPLATE, find the closing </script> tag (the last one before </body>)
# Replace:
#         updateModeUI('groq');
#     </script>
# With:
# ─────────────────────────────────────────────────────────────────────────────

CHANGE_4_FIND = "        updateModeUI('groq');\n    </script>"

CHANGE_4_REPLACE = """        updateModeUI('groq');

        async function generateReport() {
            const btn = document.getElementById('report-btn');
            btn.textContent = '⏳ Generating...';
            btn.disabled = true;
            btn.style.opacity = '0.7';
            try {
                const res = await fetch('/api/report/generate', { method: 'POST' });
                const data = await res.json();
                if (data.status === 'ok') {
                    btn.textContent = '✅ Report Ready';
                    btn.style.background = '#16a34a';
                    // Open PDF in new tab
                    window.open('/reports/' + data.report, '_blank');
                    // Also log screenshot name
                    if (data.screenshot) {
                        console.log('[WASP] Screenshot saved:', data.screenshot);
                    }
                    setTimeout(() => {
                        btn.textContent = '📄 Generate Report';
                        btn.style.background = '#dc2626';
                        btn.disabled = false;
                        btn.style.opacity = '1';
                    }, 4000);
                } else {
                    btn.textContent = '❌ Error';
                    btn.style.background = '#7c3aed';
                    btn.disabled = false;
                    btn.style.opacity = '1';
                    setTimeout(() => { btn.textContent = '📄 Generate Report'; btn.style.background='#dc2626'; }, 3000);
                    alert('Report error: ' + (data.message || 'unknown'));
                }
            } catch(e) {
                btn.textContent = '❌ Failed';
                btn.disabled = false;
                btn.style.opacity = '1';
                setTimeout(() => { btn.textContent = '📄 Generate Report'; btn.style.background='#dc2626'; }, 3000);
                console.error('Report error:', e);
            }
        }
    </script>"""


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-APPLY SCRIPT
# Run this file directly to patch wasp_backend.py automatically:
#   python wasp_patch.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import re

    backend_path = "wasp_backend.py"
    with open(backend_path, "r") as f:
        src = f.read()

    original = src

    # Change 1
    if CHANGE_1_INSERT.strip() not in src:
        src = src.replace(
            CHANGE_1_AFTER,
            CHANGE_1_AFTER + "\n" + CHANGE_1_INSERT
        )
        print("[PATCH] Change 1 applied: import report_generator")
    else:
        print("[PATCH] Change 1 already applied")

    # Change 2
    if "api_generate_report" not in src:
        src = src.replace(
            CHANGE_2_BEFORE,
            CHANGE_2_INSERT + "\n" + CHANGE_2_BEFORE
        )
        print("[PATCH] Change 2 applied: /api/report/generate endpoint")
    else:
        print("[PATCH] Change 2 already applied")

    # Change 3
    if CHANGE_3_FIND in src:
        src = src.replace(CHANGE_3_FIND, CHANGE_3_REPLACE)
        print("[PATCH] Change 3 applied: Generate Report button in header")
    else:
        print("[PATCH] Change 3 — could not find target, check HTML_TEMPLATE header-right block")

    # Change 4
    if CHANGE_4_FIND in src:
        src = src.replace(CHANGE_4_FIND, CHANGE_4_REPLACE)
        print("[PATCH] Change 4 applied: generateReport() JS function")
    else:
        print("[PATCH] Change 4 — could not find target, check closing script tag")

    if src != original:
        # Backup
        with open(backend_path + ".bak", "w") as f:
            f.write(original)
        print(f"[PATCH] Backup saved to {backend_path}.bak")

        with open(backend_path, "w") as f:
            f.write(src)
        print(f"[PATCH] {backend_path} updated successfully.")
    else:
        print("[PATCH] No changes written (all already applied or targets not found).")
