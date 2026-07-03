import os, sys, subprocess, threading, queue, pickle, sqlite3, json, webbrowser
from flask import Flask, Response, jsonify, send_from_directory

app = Flask(__name__, static_folder=".", template_folder=".")

BASE         = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(BASE, "controller_part1.py")
EVAL_SCRIPT  = os.path.join(BASE, "controller_evaluate.py")
DB_TRAIN     = os.path.join(BASE, "v2v_routing_logs.db")
STATS_PKL    = os.path.join(BASE, "training_stats.pkl")

log_queue  = queue.Queue()
proc_state = {"running": False, "mode": None, "proc": None}


def stream_process(script_path, mode):
    proc_state["running"] = True
    proc_state["mode"]    = mode
    log_queue.put(f"__MODE__{mode}")
    log_queue.put(f">> Launching {os.path.basename(script_path)} ...")
    try:
        # ✅ FIX 1: Pass full environment so SUMO_HOME and PATH are available
        env = os.environ.copy()
        env["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
        env["PYTHONUTF8"] = "1"  # ✅ Forces UTF-8 output on Windows (fixes ✓ encoding error)

        # ✅ FIX 2: Add SUMO bin to PATH so sumo-gui.exe is always found
        sumo_bin = r"C:\Program Files (x86)\Eclipse\Sumo\bin"
        env["PATH"] = sumo_bin + os.pathsep + env.get("PATH", "")

        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=BASE,
            env=env,
            # ✅ FIX 3: creationflags=0 ensures the subprocess can open GUI windows
            # Do NOT use CREATE_NO_WINDOW here — that would suppress the SUMO GUI
        )
        proc_state["proc"] = proc
        for line in proc.stdout:
            log_queue.put(line.rstrip())
        proc.wait()
        log_queue.put(f">> Finished — exit code {proc.returncode}")
    except Exception as e:
        log_queue.put(f">> ERROR: {e}")
    finally:
        proc_state["running"] = False
        proc_state["mode"]    = None
        proc_state["proc"]    = None
        log_queue.put("__DONE__")


@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/train", methods=["POST"])
def train():
    if proc_state["running"]:
        return jsonify({"error": "Simulation already running."}), 400
    threading.Thread(target=stream_process, args=(TRAIN_SCRIPT, "train"), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/evaluate", methods=["POST"])
def evaluate():
    if proc_state["running"]:
        return jsonify({"error": "Simulation already running."}), 400
    threading.Thread(target=stream_process, args=(EVAL_SCRIPT, "eval"), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop():
    p = proc_state.get("proc")
    if p and proc_state["running"]:
        p.terminate()
        return jsonify({"ok": True})
    return jsonify({"error": "Nothing running."}), 400

@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                line = log_queue.get(timeout=30)
                yield f"data: {json.dumps(line)}\n\n"
                if line == "__DONE__":
                    break
            except queue.Empty:
                yield "data: __HEARTBEAT__\n\n"
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/status")
def status():
    return jsonify({"running": proc_state["running"], "mode": proc_state["mode"]})

@app.route("/stats")
def stats():
    result = {
        "has_data": False,
        "rewards": [], "epsilon": None, "training_steps": 0,
        "route_changes": [], "congestion": [], "edge_perf": []
    }
    if os.path.exists(STATS_PKL):
        with open(STATS_PKL, "rb") as f:
            s = pickle.load(f)
        rewards = s.get("episode_rewards", [])
        step = max(1, len(rewards) // 300)
        result["rewards"]        = rewards[::step]
        result["epsilon"]        = round(s.get("final_epsilon", 1.0), 4)
        result["training_steps"] = s.get("training_steps", 0)
        result["route_changes"]  = [
            {"step": c["step"], "vid": c["vid"],
             "old": ",".join(c["old"]) if isinstance(c["old"], list) and c["old"] else str(c.get("old","NONE")),
             "new": ",".join(c["new"]) if isinstance(c["new"], list) else str(c.get("new","")),
             "reason": c.get("reason", "")}
            for c in s.get("route_changes", [])[:30]
        ]
        result["has_data"] = True

    if os.path.exists(DB_TRAIN):
        conn = sqlite3.connect(DB_TRAIN)
        rows = conn.execute(
            "SELECT congestion_level, COUNT(*) FROM congestion_log GROUP BY congestion_level"
        ).fetchall()
        result["congestion"] = [{"level": r[0], "count": r[1]} for r in rows]
        rows = conn.execute(
            "SELECT edge_id, AVG(vehicle_count), AVG(avg_speed) "
            "FROM congestion_log GROUP BY edge_id ORDER BY AVG(vehicle_count) DESC"
        ).fetchall()
        result["edge_perf"] = [
            {"edge": r[0], "avg_count": round(r[1], 2), "avg_speed": round(r[2], 2)}
            for r in rows
        ]
        conn.close()
        result["has_data"] = True

    return jsonify(result)
@app.route("/eval_stats")
def eval_stats():
    db_eval = os.path.join(BASE, "v2v_eval_logs.db")
    result = {
        "has_data"         : False,
        "avg_speed"        : 0,
        "min_speed"        : 0,
        "max_speed"        : 0,
        "total_vehicles"   : 0,
        "close_events"     : 0,
        "brake_events"     : 0,
        "free_events"      : 0,
        "quality"          : "NO DATA",
        "safety_score"     : 0,
        "efficiency_score" : 0,
        "speed_over_time"  : [],
    }
    if not os.path.exists(db_eval):
        return jsonify(result)
    try:
        conn = sqlite3.connect(db_eval)
        row = conn.execute(
            "SELECT AVG(speed), MIN(speed), MAX(speed), COUNT(DISTINCT veh_id) "
            "FROM vehicle_log"
        ).fetchone()
        if row and row[0] is not None:
            result["avg_speed"]      = round(row[0], 2)
            result["min_speed"]      = round(row[1], 2)
            result["max_speed"]      = round(row[2], 2)
            result["total_vehicles"] = row[3]
        close = conn.execute(
            "SELECT COUNT(*) FROM vehicle_log WHERE warning LIKE '%CLOSE%'"
        ).fetchone()[0]
        brake = conn.execute(
            "SELECT COUNT(*) FROM vehicle_log WHERE warning LIKE '%BRAKE%'"
        ).fetchone()[0]
        free = conn.execute(
            "SELECT COUNT(*) FROM vehicle_log WHERE warning LIKE '%Free%'"
        ).fetchone()[0]
        result["close_events"] = close
        result["brake_events"] = brake
        result["free_events"]  = free
        rows = conn.execute(
            "SELECT step, AVG(speed) FROM vehicle_log GROUP BY step ORDER BY step"
        ).fetchall()
        step_val = max(1, len(rows) // 200)
        result["speed_over_time"] = [
            {"step": r[0], "speed": round(r[1], 2)}
            for r in rows[::step_val]
        ]
        total_events = close + brake + free
        if total_events > 0:
            result["safety_score"]     = round((free / total_events) * 100, 1)
            result["efficiency_score"] = round((result["avg_speed"] / 20.0) * 100, 1)
        s = result["safety_score"]
        e = result["efficiency_score"]
        if s >= 80 and e >= 70:
            result["quality"] = "EXCELLENT"
        elif s >= 65 and e >= 55:
            result["quality"] = "GOOD"
        elif s >= 45:
            result["quality"] = "FAIR"
        else:
            result["quality"] = "POOR"
        result["has_data"] = True
        conn.close()
    except Exception as ex:
        result["error"] = str(ex)
    return jsonify(result)

if __name__ == "__main__":
    print("=" * 52)
    print("  V2V Traffic AI — Backend running")
    print("  Open:  http://localhost:5000")
    print("=" * 52)
    # Auto-open browser after 1.5 seconds
    threading.Timer(1.5, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(debug=False, threaded=True, port=5000)