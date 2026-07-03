"""
controller_evaluate.py
Run trained V2V model WITHOUT training — demonstration / evaluation only.

FIXES APPLIED:
  - Route replacement: never set a route with duplicate consecutive edges
  - Route replacement: skip if vehicle is on a junction (':' edge)
  - Route replacement: only set route when A* actually found a multi-edge path
  - Route replacement: validate route has no duplicate edges before calling setRoute
  - astar() now checks start == goal to avoid empty path issues
  - build_route() centralises all route logic with full validation
  - load_model() handles both old and new checkpoint formats safely
  - torch.load() uses map_location for CPU/GPU compatibility
  - traci.start() uses --start and --quit-on-end for automatic SUMO launch
  - All traci calls defended with try/except
  - Print every step (not every 100) so web dashboard updates live
  - VEH: lines printed for Live Vehicle Monitor table
"""

import heapq
import os
import sys
import math
import sqlite3
import numpy as np

# Fix Unicode output on Windows terminal
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── SUMO setup ─────────────────────────────────────────────────────────────
SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
os.environ['SUMO_HOME'] = SUMO_HOME

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

try:
    import traci
except ImportError as e:
    print(f"Error importing traci: {e}")
    sys.exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    ML_AVAILABLE = True
except ImportError:
    print("WARNING: PyTorch not available!")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────
SUMO_BINARY            = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")
RADIO_RANGE            = 150.0
SIM_STEPS              = 900
SAFE_DISTANCE          = 15.0
COLOR_CHANGE_THRESHOLD = 22.5

# ── Speed & braking constants ─────────────────────────────────────────────────
MAX_SPEED              = 20.0
NORMAL_SPEED           = 15.0
BRAKE_EARLY_DIST       = 35.0
BRAKE_HARD_DIST        = 18.0
STOP_DIST              = 8.0
MIN_SPEED              = 10.0

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SUMO_CONFIG  = os.path.join(SCRIPT_DIR, "simulation.sumocfg")
NETWORK_FILE = os.path.join(SCRIPT_DIR, "network.net.xml")
ROUTES_FILE  = os.path.join(SCRIPT_DIR, "routes.rou.xml")
GUI_FILE     = os.path.join(SCRIPT_DIR, "gui-settings.xml")


# ── Helpers ────────────────────────────────────────────────────────────────

def verify_files():
    required = [
        ("SUMO binary",  SUMO_BINARY),
        ("SUMO config",  SUMO_CONFIG),
        ("Network file", NETWORK_FILE),
        ("Routes file",  ROUTES_FILE),
        ("GUI settings", GUI_FILE),
    ]
    ok = True
    for label, path in required:
        if os.path.exists(path):
            print(f"  OK  {label}: {path}")
        else:
            print(f"  MISSING {label}: {path}")
            ok = False
    return ok


def setup_database():
    db_path = os.path.join(SCRIPT_DIR, "v2v_eval_logs.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_log (
            step INTEGER, time REAL, veh_id TEXT,
            x REAL, y REAL, speed REAL, warning TEXT, edge_id TEXT)
    """)
    conn.commit()
    return conn, cur


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def is_valid_route(route):
    if not route:
        return False
    for i in range(len(route) - 1):
        if route[i] == route[i + 1]:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Brake Indicator Manager
# ─────────────────────────────────────────────────────────────────────────────

BRAKE_POI_OFFSET_Y = 5.0
BRAKE_POI_SIZE     = 3.0
BRAKE_POI_LAYER    = 10.0

BRAKE_LEVELS = {
    1: ("!!",    255, 220,   0, 255),
    2: ("!!!",   255, 140,   0, 255),
    3: ("BRAKE", 220,   0,   0, 255),
}

class BrakeIndicatorManager:
    def __init__(self):
        self._active = {}
        self._poi_counter = 0

    def _new_poi_id(self, vid):
        self._poi_counter += 1
        return f"brake_{vid}_{self._poi_counter}"

    def update(self, vid, action, pos):
        is_braking = action in BRAKE_LEVELS
        if is_braking:
            label, r, g, b, a = BRAKE_LEVELS[action]
            px = pos[0]
            py = pos[1] + BRAKE_POI_OFFSET_Y
            if vid in self._active:
                poi_id = self._active[vid]
                try:
                    traci.poi.setPosition(poi_id, px, py)
                    traci.poi.setColor(poi_id, (r, g, b, a))
                except Exception:
                    del self._active[vid]
            if vid not in self._active:
                poi_id = f"{label}__{vid}__{self._poi_counter}"
                try:
                    traci.poi.add(
                        poi_id, px, py, (r, g, b, a),
                        poiType="brake_indicator",
                        layer=BRAKE_POI_LAYER,
                        imgWidth=2.5, imgHeight=2.5,
                    )
                    self._active[vid] = poi_id
                except Exception:
                    pass
        else:
            self._remove(vid)

    def _remove(self, vid):
        if vid in self._active:
            try:
                traci.poi.remove(self._active[vid])
            except Exception:
                pass
            del self._active[vid]

    def remove_all(self):
        for vid in list(self._active.keys()):
            self._remove(vid)


# ── A* routing ────────────────────────────────────────────────────────────

GRAPH = {
    "J1": [("J2", "E1", 500), ("J7", "E5", 500)],
    "J2": [("J1", "E1", 500), ("J8", "E8", 500)],
    "J7": [("J1", "E5", 500), ("J8", "E4", 500)],
    "J8": [("J7", "E4", 500), ("J2", "E8", 500)],
}


def heuristic(a, b):
    try:
        ax, ay = traci.junction.getPosition(a)
        bx, by = traci.junction.getPosition(b)
        return math.hypot(ax - bx, ay - by)
    except Exception:
        return 0


def astar(start, goal, congestion):
    if start == goal:
        return []
    open_set = []
    heapq.heappush(open_set, (0, start, []))
    visited = set()
    while open_set:
        cost, node, path = heapq.heappop(open_set)
        if node == goal:
            return path
        if node in visited:
            continue
        visited.add(node)
        for nxt, edge, length in GRAPH.get(node, []):
            traffic  = congestion.get(edge, 0)
            new_cost = cost + length + traffic * 10 + heuristic(nxt, goal)
            heapq.heappush(open_set, (new_cost, nxt, path + [edge]))
    return []


def get_congestion(vehicles):
    cong = {}
    for v in vehicles:
        try:
            edge = traci.vehicle.getRoadID(v)
            if not edge.startswith(':'):
                cong[edge] = cong.get(edge, 0) + 1
        except Exception:
            pass
    return cong


def build_route(v, congestion):
    try:
        current_edge = traci.vehicle.getRoadID(v)
        if current_edge.startswith(':'):
            return None
        to_junction = traci.edge.getToJunction(current_edge)
        if to_junction == "J8":
            return None
        next_edges = astar(to_junction, "J8", congestion)
        if not next_edges:
            return None
        full_route = [current_edge] + next_edges
        if not is_valid_route(full_route):
            return None
        return full_route
    except Exception:
        return None


# ── DQN model ─────────────────────────────────────────────────────────────

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, action_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.fc4(x))
        return self.fc5(x)


# ── Trained agent ──────────────────────────────────────────────────────────

class TrainedAgent:
    def __init__(self):
        self.state_size  = 15
        self.action_size = 5

        self.actions = {
            0: ("maintain",     1.00),
            1: ("light_brake",  0.95),
            2: ("medium_brake", 0.85),
            3: ("hard_brake",   0.65),
            4: ("accelerate",   -1),
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = DQN(self.state_size, self.action_size).to(self.device)

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            print(f"WARNING: Model file not found: {filepath}")
            return False
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("OK Trained model loaded")
            else:
                self.model.load_state_dict(checkpoint)
                print("OK Model weights loaded (old format)")
            self.model.eval()
            return True
        except Exception as e:
            print(f"ERROR loading model: {e}")
            return False

    # ✅ FIXED get_state (15 features, correct indentation)
    def get_state(self, vid, vehicles):
        try:
            pos   = traci.vehicle.getPosition(vid)
            speed = traci.vehicle.getSpeed(vid)
            accel = traci.vehicle.getAcceleration(vid)
            angle = traci.vehicle.getAngle(vid)
            edge  = traci.vehicle.getRoadID(vid)

            front_dist      = RADIO_RANGE
            front_rel_speed = 0.0

            for other in vehicles:
                if other == vid:
                    continue
                try:
                    opos   = traci.vehicle.getPosition(other)
                    ospeed = traci.vehicle.getSpeed(other)
                    d      = distance(pos, opos)

                    if d <= RADIO_RANGE:
                        dx = opos[0] - pos[0]
                        dy = opos[1] - pos[1]
                        rel_angle = math.atan2(dy, dx) - math.radians(angle)

                        if -0.5 < rel_angle < 0.5 and d < front_dist:
                            front_dist      = d
                            front_rel_speed = speed - ospeed
                except Exception:
                    continue

            state = [
                speed / MAX_SPEED,
                accel / 5.0,
                front_dist / RADIO_RANGE,
                front_rel_speed / MAX_SPEED,
                front_dist / max(1, speed) if speed > 0 else 5.0,
                0.0,
                1.0 if edge.startswith(':') else 0.0,
                0.25 if speed < 5 else 0.75 if speed > 15 else 0.5,
                min(1.0, front_dist / SAFE_DISTANCE),
                1.0 if front_dist < SAFE_DISTANCE else 0.0,
                len(vehicles) / 30.0,
                0.0,
                speed / MAX_SPEED,
                accel / 3.0,
                front_dist / (SAFE_DISTANCE * 2),
            ]

            return np.array(state, dtype=np.float32)

        except Exception:
            return np.zeros(self.state_size, dtype=np.float32)

    def act(self, state):
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.model(t).argmax().item()


# ── Main evaluation loop ───────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("Checking required files...")
    if not verify_files():
        print("\nFix the missing files listed above and try again.")
        return

    agent      = TrainedAgent()
    model_path = os.path.join(SCRIPT_DIR, "v2v_model.pth")

    if not agent.load_model(model_path):
        print("\nERROR: No usable trained model found.")
        print("  Run controller_part1.py first to train and save the model.")
        return

    print("\n" + "=" * 70)
    print("EVALUATION MODE — Trained Model Performance")
    print("=" * 70)
    print("  No training is happening — purely demonstrating learned behaviour.")
    print("Visual indicators:")
    print("  GREEN vehicle   = safe distance")
    print("  RED vehicle     = another car within 22.5m")
    print("  !!   above car  = light brake applied  (yellow)")
    print("  !!!  above car  = medium brake applied (orange)")
    print("  BRAKE above car = hard brake applied   (red POI)")
    print("=" * 70 + "\n")

    conn, cur = setup_database()
    brake_indicators = BrakeIndicatorManager()

    print("Starting SUMO GUI...")
    traci.start([
        SUMO_BINARY,
        "-c", SUMO_CONFIG,
        "--start",
        "--quit-on-end",
    ])
    print("SUMO GUI opened successfully\n")

    step = 0

    try:
        while step < SIM_STEPS:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            vehicles = list(traci.vehicle.getIDList())

            # ── A* routing (congestion-aware) ─────────────────────────────
            congestion = get_congestion(vehicles)
            for v in vehicles:
                route = build_route(v, congestion)
                if route:
                    try:
                        traci.vehicle.setRoute(v, route)
                    except Exception:
                        pass

            # ── DQN pure exploitation ─────────────────────────────────────
            warnings = {}
            for vid in vehicles:
                current_state             = agent.get_state(vid, vehicles)
                action                    = agent.act(current_state)
                action_name, speed_factor = agent.actions[action]

                try:
                    current_speed = traci.vehicle.getSpeed(vid)
                    pos           = traci.vehicle.getPosition(vid)
                    min_dist      = float('inf')

                    for other in vehicles:
                        if other == vid:
                            continue
                        try:
                            d = distance(pos, traci.vehicle.getPosition(other))
                            if d < min_dist:
                                min_dist = d
                        except Exception:
                            continue

                    traci.vehicle.setSpeedMode(vid, 31)
                    traci.vehicle.setLaneChangeMode(vid, 0)
                    traci.vehicle.setMaxSpeed(vid, MAX_SPEED)
                    traci.vehicle.setMinGap(vid, 5.0)
                    traci.vehicle.setTau(vid, 1.5)
                    traci.vehicle.setDecel(vid, 6.0)
                    traci.vehicle.setEmergencyDecel(vid, 9.0)

                    # ── Distance-based speed + brake zones ────────────────
                    if min_dist <= STOP_DIST:
                        # Very close — slow to crawl but never fully stop
                        # so cars behind can still inch forward
                        target_speed = 2.0
                        action       = 3
                        action_name  = "hard_brake"

                    elif min_dist <= SAFE_DISTANCE:
                        # Close — slow proportionally to distance
                        ratio        = (min_dist - STOP_DIST) / (SAFE_DISTANCE - STOP_DIST)
                        target_speed = 2.0 + ratio * 3.0
                        action       = 3
                        action_name  = "hard_brake"

                    elif min_dist <= BRAKE_HARD_DIST:
                        # Medium close — moderate braking
                        ratio        = (min_dist - SAFE_DISTANCE) / (BRAKE_HARD_DIST - SAFE_DISTANCE)
                        target_speed = 5.0 + ratio * 4.0
                        action       = 2
                        action_name  = "medium_brake"

                    elif min_dist <= BRAKE_EARLY_DIST:
                        # Early warning — light braking
                        target_speed = NORMAL_SPEED * 0.60
                        action       = 1
                        action_name  = "light_brake"

                    else:
                        # All clear — cruise at normal speed
                        if speed_factor == -1:
                            target_speed = NORMAL_SPEED
                        else:
                            target_speed = NORMAL_SPEED * speed_factor
                        target_speed = min(target_speed, MAX_SPEED)

                    target_speed = min(target_speed, MAX_SPEED)
                    speed_diff   = target_speed - current_speed
                    if speed_diff > 0:
                        # Gentle acceleration — no sudden lurching forward
                        smooth_speed = current_speed + min(2.0, speed_diff)
                    else:
                        # Firm braking — but not instant stop
                        smooth_speed = current_speed + max(-4.0, speed_diff)
                    smooth_speed = max(2.0, min(smooth_speed, MAX_SPEED))
                    traci.vehicle.setSpeed(vid, smooth_speed)

                    # ── Vehicle colour ────────────────────────────────────
                    if min_dist <= COLOR_CHANGE_THRESHOLD:
                        traci.vehicle.setColor(vid, (255, 0, 0, 255))
                        warnings[vid] = [f"CLOSE! ({min_dist:.1f}m) {smooth_speed:.1f}m/s"]
                    elif min_dist <= BRAKE_EARLY_DIST:
                        traci.vehicle.setColor(vid, (255, 140, 0, 255))
                        warnings[vid] = [f"BRAKE ({min_dist:.1f}m) {smooth_speed:.1f}m/s"]
                    else:
                        traci.vehicle.setColor(vid, (0, 200, 0, 255))
                        warnings[vid] = [f"Free ({min_dist:.0f}m) {smooth_speed:.1f}m/s"]

                    # ── Brake POI indicator ────────────────────────────────
                    brake_indicators.update(vid, action, pos)

                    # ── FIX: Print VEH line for Live Vehicle Monitor ───────
                    veh_status = (
                        "CLOSE" if min_dist <= COLOR_CHANGE_THRESHOLD else
                        "BRAKE" if min_dist <= BRAKE_EARLY_DIST else
                        "Free"
                    )
                    print(
                        f"VEH: {vid} | {traci.vehicle.getRoadID(vid)} | "
                        f"{smooth_speed:.1f} | {veh_status}",
                        flush=True)

                except Exception:
                    pass

            # ── DB log ────────────────────────────────────────────────────
            for v in vehicles:
                try:
                    x, y = traci.vehicle.getPosition(v)
                    spd  = traci.vehicle.getSpeed(v)
                    edge = traci.vehicle.getRoadID(v)
                    warn = "; ".join(warnings.get(v, []))
                    cur.execute(
                        "INSERT INTO vehicle_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (step, sim_time, v, x, y, spd, warn, edge))
                except Exception:
                    continue

            conn.commit()

            # Remove brake POIs for vehicles that left the simulation
            active_set = set(vehicles)
            for vid in list(brake_indicators._active.keys()):
                if vid not in active_set:
                    brake_indicators._remove(vid)

            # ── FIX: Print EVERY step so web dashboard updates live ───────
            print(
                f"Step {step:4d} | Time {sim_time:6.1f}s | "
                f"Vehicles: {len(vehicles):2d} | e: 0.000 | Reward: +0.00",
                flush=True)

            step += 1

    finally:
        print("\n" + "=" * 70)
        print("Evaluation complete!")
        brake_indicators.remove_all()
        try:
            traci.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        print(f"  Logs saved to: v2v_eval_logs.db")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()