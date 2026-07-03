"""
controller_part1.py
Advanced V2V Controller with:
- Training-based V2V communication
- Dynamic route selection based on congestion
- Real-time traffic analysis
- Balanced route optimization

BRAKE VISUAL INDICATOR:
  When a vehicle brakes due to another car coming close, a floating POI
  symbol appears ABOVE the vehicle in SUMO showing the brake intensity:
    - LIGHT brake  (action 1): yellow  !! label
    - MEDIUM brake (action 2): orange  !!! label  
    - HARD brake   (action 3): red     BRAKE label
  The POI disappears as soon as the vehicle stops braking.
  Color red is still used for closeness — brake indicator is SEPARATE.
"""

import heapq
import os
import sys
import math
import sqlite3
import numpy as np
import random
from collections import deque, defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── SUMO setup ────────────────────────────────────────────────────────────────
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
    import torch.optim as optim
    import torch.nn.functional as F
    ML_AVAILABLE = True
    print("✓ PyTorch available - Using DQN training")
except ImportError:
    print("WARNING: PyTorch not available - Using simple fallback")
    ML_AVAILABLE = False

# ── Simulation constants ──────────────────────────────────────────────────────
SUMO_BINARY            = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")
RADIO_RANGE            = 150.0
SIM_STEPS              = 5000
SAFE_DISTANCE          = 15.0      # hard stop zone — emergency brake
COLOR_CHANGE_THRESHOLD = 22.5      # vehicle turns RED at this distance
CONGESTION_THRESHOLD   = 3
ROUTE_UPDATE_INTERVAL  = 50
MAX_ROUTE_LENGTH       = 5

# ── Speed & braking constants ─────────────────────────────────────────────────
MAX_SPEED              = 20.0      # ~180 km/h — max allowed speed
NORMAL_SPEED           = 15.0      # ~144 km/h — fast cruising speed
BRAKE_EARLY_DIST       = 35.0      # start light braking when another car is within 35m
BRAKE_HARD_DIST        = 18.0      # hard brake when within 18m
STOP_DIST              = 8.0       # near-stop when within 8m (prevent overlap)
MIN_SPEED              = 10.0       # never go below this (avoid SUMO teleporting)

# ── Brake POI settings ────────────────────────────────────────────────────────
# These control how the floating brake indicator looks above each vehicle
BRAKE_POI_OFFSET_Y   = 5.0    # metres above vehicle centre
BRAKE_POI_SIZE       = 3.0    # size of the POI circle in SUMO
BRAKE_POI_LAYER      = 10.0   # draw on top of everything

# Brake level definitions:  (label_text, R, G, B, A)
BRAKE_LEVELS = {
    1: ("!!",    255, 220,   0, 255),   # light  — yellow
    2: ("!!!",   255, 140,   0, 255),   # medium — orange
    3: ("BRAKE", 220,   0,   0, 255),   # hard   — deep red (different from vehicle red)
}

# ── File paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SUMO_CONFIG  = os.path.join(SCRIPT_DIR, "simulation.sumocfg")
NETWORK_FILE = os.path.join(SCRIPT_DIR, "network.net.xml")
ROUTES_FILE  = os.path.join(SCRIPT_DIR, "routes.rou.xml")
GUI_FILE     = os.path.join(SCRIPT_DIR, "gui-settings.xml")


# ─────────────────────────────────────────────────────────────────────────────
# Brake Indicator Manager
# Manages floating POI symbols above braking vehicles in SUMO GUI
# ─────────────────────────────────────────────────────────────────────────────

class BrakeIndicatorManager:
    """
    Shows a floating visual indicator above a vehicle when it brakes.

    How it works:
      - When a vehicle brakes, a POI (Point of Interest) is created
        just above the vehicle's position in SUMO.
      - The POI is coloured and labelled based on brake intensity:
          action 1 (light)  → yellow  !!
          action 2 (medium) → orange  !!!
          action 3 (hard)   → red     BRAKE
      - Every step the POI moves with the vehicle.
      - When the vehicle stops braking the POI is removed.
      - This is completely separate from the red/green vehicle colour
        which shows closeness.
    """

    def __init__(self):
        # vid -> poi_id currently shown for that vehicle
        self._active = {}
        self._poi_counter = 0

    def _new_poi_id(self, vid):
        self._poi_counter += 1
        return f"brake_{vid}_{self._poi_counter}"

    def update(self, vid, action, pos):
        """
        Call every simulation step for every vehicle.
        action  : int  — the DQN action (0-4)
        pos     : (x, y) — current vehicle position from traci
        """
        is_braking = action in BRAKE_LEVELS   # actions 1, 2, 3

        if is_braking:
            label, r, g, b, a = BRAKE_LEVELS[action]
            px = pos[0]
            py = pos[1] + BRAKE_POI_OFFSET_Y

            if vid in self._active:
                # Move existing POI to follow the vehicle
                poi_id = self._active[vid]
                try:
                    traci.poi.setPosition(poi_id, px, py)
                    traci.poi.setColor(poi_id, (r, g, b, a))
                except Exception:
                    # POI disappeared for some reason — recreate it
                    del self._active[vid]

            if vid not in self._active:
                # Create a new POI above this vehicle.
                # IMPORTANT: In SUMO GUI the POI "id" string is what gets
                # displayed as the visible label when "show poi name" is on.
                # We use the brake symbol as the POI id so it shows clearly.
                poi_id = f"{label}__{vid}__{self._poi_counter}"
                try:
                    traci.poi.add(
                        poi_id,          # this text IS the visible label in SUMO
                        px, py,
                        (r, g, b, a),
                        poiType="brake_indicator",
                        layer=BRAKE_POI_LAYER,
                        imgWidth=2.5,    # visible coloured dot size
                        imgHeight=2.5,
                    )
                    self._active[vid] = poi_id
                except Exception:
                    pass

        else:
            # Vehicle not braking — remove its POI if one exists
            self._remove(vid)

    def _remove(self, vid):
        if vid in self._active:
            try:
                traci.poi.remove(self._active[vid])
            except Exception:
                pass
            del self._active[vid]

    def remove_all(self):
        """Clean up all POIs — call at end of simulation."""
        for vid in list(self._active.keys()):
            self._remove(vid)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    db_path = os.path.join(SCRIPT_DIR, "v2v_routing_logs.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_log (
            step INTEGER, time REAL, veh_id TEXT,
            x REAL, y REAL, speed REAL, warning TEXT, edge_id TEXT)
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS congestion_log (
            step INTEGER, time REAL, edge_id TEXT,
            vehicle_count INTEGER, avg_speed REAL, congestion_level TEXT)
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS route_change_log (
            step INTEGER, time REAL, veh_id TEXT,
            old_route TEXT, new_route TEXT, reason TEXT)
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
# Congestion Analyser
# ─────────────────────────────────────────────────────────────────────────────

class CongestionAnalyzer:
    def __init__(self):
        self.history           = defaultdict(list)
        self.congestion_levels = {}

    def update(self, step, vehicles):
        edge_data = defaultdict(lambda: {'count': 0, 'speeds': []})
        for vid in vehicles:
            try:
                edge = traci.vehicle.getRoadID(vid)
                if edge.startswith(':'):
                    continue
                speed = traci.vehicle.getSpeed(vid)
                edge_data[edge]['count'] += 1
                edge_data[edge]['speeds'].append(speed)
            except Exception:
                continue

        self.congestion_levels.clear()
        for edge, data in edge_data.items():
            count     = data['count']
            avg_speed = float(np.mean(data['speeds'])) if data['speeds'] else 0.0
            if count >= CONGESTION_THRESHOLD * 2:
                level = "HIGH"
            elif count >= CONGESTION_THRESHOLD:
                level = "MEDIUM"
            else:
                level = "LOW"
            self.congestion_levels[edge] = {
                'count': count, 'avg_speed': avg_speed, 'level': level}
            self.history[edge].append((step, count, avg_speed))
            if len(self.history[edge]) > 100:
                self.history[edge].pop(0)

    def get_congestion_score(self, edge):
        if edge not in self.congestion_levels:
            return 0
        data = self.congestion_levels[edge]
        return data['count'] * 10 + max(0, (20 - data['avg_speed'])) * 2

    def get_edge_info(self, edge):
        return self.congestion_levels.get(edge, {
            'count': 0, 'avg_speed': 20.0, 'level': 'LOW'})

    def get_congestion_report(self):
        report = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
        for edge, data in self.congestion_levels.items():
            report[data['level']].append(edge)
        return report


# ─────────────────────────────────────────────────────────────────────────────
# A* path planning
# ─────────────────────────────────────────────────────────────────────────────

GRAPH = {
    "J1":  [("J2", "E1", 500), ("J7", "E5", 500)],
    "J2":  [("J1", "E1", 500), ("J8", "E8", 500)],
    "J7":  [("J1", "E5", 500), ("J8", "E4", 500)],
    "J8":  [("J7", "E4", 500), ("J2", "E8", 500)],
    "J3":  [("J4", "E2", 500)],
    "J4":  [("J3", "E2", 500)],
    "J5":  [("J6", "E3", 500)],
    "J6":  [("J5", "E3", 500)],
    "J9":  [("J10","E6", 500)],
    "J10": [("J9", "E6", 500)],
    "J11": [("J12","E7", 500)],
    "J12": [("J11","E7", 500)],
}


def heuristic(a, b):
    try:
        ax, ay = traci.junction.getPosition(a)
        bx, by = traci.junction.getPosition(b)
        return math.hypot(ax - bx, ay - by)
    except Exception:
        return 0


def astar_with_congestion(start, goal, congestion_analyzer):
    if start == goal:
        return [], 0
    open_set = []
    heapq.heappush(open_set, (0, start, []))
    visited = set()
    while open_set:
        cost, node, path = heapq.heappop(open_set)
        if node == goal:
            return path, cost
        if node in visited:
            continue
        visited.add(node)
        for nxt, edge, length in GRAPH.get(node, []):
            cong_score = congestion_analyzer.get_congestion_score(edge)
            h_cost     = heuristic(nxt, goal)
            new_cost   = cost + length + cong_score * 5 + h_cost
            heapq.heappush(open_set, (new_cost, nxt, path + [edge]))
    return [], float('inf')


def get_alternative_routes(start, goal, congestion_analyzer, num_routes=3):
    routes = []
    best_route, best_cost = astar_with_congestion(start, goal, congestion_analyzer)
    if best_route:
        routes.append({'edges': best_route, 'cost': best_cost, 'type': 'optimal'})
    temp_penalties = {}
    for _ in range(num_routes - 1):
        for route in routes:
            for edge in route['edges']:
                temp_penalties[edge] = temp_penalties.get(edge, 0) + 100
        alt_route, alt_cost = astar_with_congestion(start, goal, congestion_analyzer)
        if alt_route and alt_route not in [r['edges'] for r in routes]:
            routes.append({
                'edges': alt_route,
                'cost':  alt_cost + sum(temp_penalties.get(e, 0) for e in alt_route),
                'type':  'alternative'})
    return sorted(routes, key=lambda x: x['cost'])


# ─────────────────────────────────────────────────────────────────────────────
# Route Manager
# ─────────────────────────────────────────────────────────────────────────────

class RouteManager:
    def __init__(self, congestion_analyzer):
        self.congestion_analyzer = congestion_analyzer
        self.vehicle_routes      = {}
        self.route_updates       = {}
        self.route_changes       = []

    def should_update_route(self, vid, step):
        return (step - self.route_updates.get(vid, -999)) >= ROUTE_UPDATE_INTERVAL

    def get_best_route(self, vid, current_edge, goal="J8"):
        try:
            end_junction = traci.edge.getToJunction(current_edge)
            if end_junction == goal:
                return None, None
            routes = get_alternative_routes(
                end_junction, goal, self.congestion_analyzer, num_routes=3)
            if not routes:
                return None, None
            best_route = routes[0]
            if not best_route['edges']:
                return None, None
            full_route = [current_edge] + best_route['edges']
            if not is_valid_route(full_route):
                return None, None
            analysis = self.analyze_route(best_route['edges'])
            return full_route, analysis
        except Exception as e:
            print(f"Route error for {vid}: {e}")
            return None, None

    def analyze_route(self, edges):
        total_vehicles = 0
        total_speed    = 0
        congestion_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for edge in edges:
            info = self.congestion_analyzer.get_edge_info(edge)
            total_vehicles += info['count']
            total_speed    += info['avg_speed']
            congestion_counts[info['level']] += 1
        avg_speed = total_speed / len(edges) if edges else 0
        return {
            'total_vehicles':    total_vehicles,
            'avg_speed':         avg_speed,
            'high_congestion':   congestion_counts['HIGH'],
            'medium_congestion': congestion_counts['MEDIUM'],
            'low_congestion':    congestion_counts['LOW'],
            'quality':           'GOOD' if congestion_counts['HIGH'] == 0 else 'POOR'
        }

    def update_vehicle_route(self, vid, step, current_edge, cur, goal="J8"):
        if not self.should_update_route(vid, step):
            return False
        new_route, analysis = self.get_best_route(vid, current_edge, goal)
        if new_route:
            old_route = self.vehicle_routes.get(vid, [])
            if new_route != old_route:
                self.vehicle_routes[vid] = new_route
                self.route_updates[vid]  = step
                reason = (f"Congestion: {analysis['quality']}, "
                          f"Vehicles: {analysis['total_vehicles']}")
                self.route_changes.append({
                    'step': step, 'vid': vid,
                    'old':  old_route, 'new': new_route, 'reason': reason})
                try:
                    cur.execute(
                        "INSERT INTO route_change_log VALUES (?, ?, ?, ?, ?, ?)",
                        (step, traci.simulation.getTime(), vid,
                         ','.join(old_route) if old_route else 'NONE',
                         ','.join(new_route), reason))
                except Exception:
                    pass
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# DQN model
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# V2V Agent
# ─────────────────────────────────────────────────────────────────────────────

class V2VAgent:
    def __init__(self, state_size=15, action_size=5):
        self.state_size   = state_size
        self.action_size  = action_size
        self.actions = {
            0: ("maintain",     1.00),
            1: ("light_brake",  0.95),
            2: ("medium_brake", 0.85),
            3: ("hard_brake",   0.65),
            4: ("accelerate",   -1),
        }
        self.gamma         = 0.95
        self.epsilon       = 0.5
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001
        self.batch_size    = 64
        self.memory        = deque(maxlen=10000)

        if ML_AVAILABLE:
            self.device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model        = DQN(state_size, action_size).to(self.device)
            self.target_model = DQN(state_size, action_size).to(self.device)
            self.optimizer    = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.update_target_model()
        else:
            self.device = self.model = self.target_model = self.optimizer = None

        self.training_step = 0
        self.congestion_analyzer = None
    def set_congestion_analyzer(self, analyzer):
        self.congestion_analyzer = analyzer    
    def update_target_model(self):
        if ML_AVAILABLE and self.model is not None:
            self.target_model.load_state_dict(self.model.state_dict())

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
            congestion_score = 0.0
            if hasattr(self, 'congestion_analyzer') and self.congestion_analyzer:
                congestion_score = self.congestion_analyzer.get_congestion_score(edge) / 100.0
            state = [
                speed / MAX_SPEED,
                accel / 5.0,
                front_dist / RADIO_RANGE,
                front_rel_speed / MAX_SPEED,
                front_dist / max(1, speed) if speed > 0 else 5.0,
                congestion_score,
                1.0 if edge.startswith(':') else 0.0,
                0.25 if speed < 5 else 0.75 if speed > 15 else 0.5,
                min(1.0, front_dist / SAFE_DISTANCE),
                1.0 if front_dist < SAFE_DISTANCE else 0.0,
                len(vehicles) / 30.0,
                self.training_step / 10000.0,
                speed / MAX_SPEED,
                accel / 3.0,
                front_dist / (SAFE_DISTANCE * 2),
            ]
            return np.array(state, dtype=np.float32)
        except Exception:
            return np.zeros(self.state_size, dtype=np.float32)

    def act(self, state):
        if not ML_AVAILABLE:
            return 2 if state[8] < 1.0 else 0
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        self.model.eval()           # ADD THIS LINE
        with torch.no_grad():
            t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.model(t).argmax().item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if not ML_AVAILABLE or len(self.memory) < self.batch_size:
            return
        self.model.train()
        batch       = random.sample(self.memory, self.batch_size)
        states      = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        actions     = torch.LongTensor([x[1] for x in batch]).to(self.device)
        rewards     = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(self.device)
        dones       = torch.FloatTensor([x[4] for x in batch]).to(self.device)

        with torch.no_grad():
            next_actions  = self.model(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_model(next_states).gather(1, next_actions).squeeze()
            target_q      = rewards + (1 - dones) * self.gamma * next_q_values

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss      = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.training_step += 1
        if self.training_step % 100 == 0:
            self.update_target_model()
    def calculate_reward(self, vid, vehicles, action_taken):
        try:
            pos    = traci.vehicle.getPosition(vid)
            speed  = traci.vehicle.getSpeed(vid)
            reward = 0.0
            if 13 <= speed <= 17:
                 reward += 4.0
            elif 10 <= speed <= 19:
                 reward += 1.0
            elif speed < 10:
                 reward -= 2.0
            elif speed > 20:
                 reward -=3.0     
            min_dist = float('inf')
            for other in vehicles:
                if other == vid:
                    continue
                try:
                    min_dist = min(min_dist, distance(pos, traci.vehicle.getPosition(other)))
                except Exception:
                    continue
            if min_dist < SAFE_DISTANCE:
                reward -= 5.0
            elif min_dist < SAFE_DISTANCE * 1.5:
                reward -= 2.0
            else:
                reward += 0.5
            if action_taken in [1, 2, 3] and min_dist > SAFE_DISTANCE * 2:
                reward -= 1.0
            try:
                if abs(traci.vehicle.getAcceleration(vid)) < 2.0:
                    reward += 0.5
            except Exception:
                pass
            return reward
        except Exception:
            return 0.0

    def save_model(self, filepath):
        if not ML_AVAILABLE or self.model is None:
            return
        try:
            torch.save({
                'model_state_dict':     self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon':              self.epsilon,
                'training_step':        self.training_step,
            }, filepath)
            print(f"✓ Model saved to {filepath}")
        except Exception as e:
            print(f"WARNING: Could not save model: {e}")

    def load_model(self, filepath):
        if not ML_AVAILABLE or self.model is None:
            return False
        if not os.path.exists(filepath):
            return False
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint and self.optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon       = float(checkpoint.get('epsilon',       self.epsilon))
                self.training_step = int(  checkpoint.get('training_step', 0))
            else:
                print("WARNING: Old checkpoint format — loading weights only.")
                self.model.load_state_dict(checkpoint)
            self.update_target_model()
            print(f"✓ Model loaded from {filepath}  "
                  f"(step={self.training_step}, epsilon={self.epsilon:.3f})")
            return True
        except RuntimeError as e:
            print(f"WARNING: Model architecture mismatch — starting fresh. ({e})")
            return False
        except Exception as e:
            print(f"WARNING: Could not load model '{filepath}' — starting fresh. ({e})")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("Checking required files...")
    if not verify_files():
        print("\nFix the missing files listed above and try again.")
        return

    conn, cur = setup_database()

    agent               = V2VAgent(state_size=15, action_size=5)
    congestion_analyzer = CongestionAnalyzer()
    route_manager       = RouteManager(congestion_analyzer)
    brake_indicators    = BrakeIndicatorManager()
    agent.set_congestion_analyzer(congestion_analyzer)       

    model_path = os.path.join(SCRIPT_DIR, "v2v_model.pth")
    if agent.load_model(model_path):
        print("→ Continuing training from saved model")
    else:
        print("→ Starting training from scratch")

    print("\n" + "=" * 70)
    print("ADVANCED V2V CONTROLLER WITH DYNAMIC ROUTING")
    print("=" * 70)
    print("Visual indicators:")
    print("  GREEN vehicle   = safe distance")
    print("  RED vehicle     = another car within 22.5m")
    print("  !!   above car  = light brake applied  (yellow)")
    print("  !!!  above car  = medium brake applied (orange)")
    print("  BRAKE above car = hard brake applied   (red POI)")
    print("=" * 70 + "\n")

    print("Starting SUMO GUI...")
    traci.start([
        SUMO_BINARY,
        "-c", SUMO_CONFIG,
        "--start",
        "--quit-on-end",
    ])
    print("SUMO GUI opened successfully\n")

    step            = 0
    vehicle_states  = {}
    episode_rewards = []
    vehicle_goals   = {}   # vid -> fixed destination junction (set once at spawn)

    # Original vehicle definitions from routes.rou.xml — used for respawning
    SPAWN_DEFS = [
        # (vid,         route_id,    departPos, departSpeed)
        ("veh_h1_1", "route_h1",  10,  24.0),
        ("veh_h1_2", "route_h1",  60,  22.0),
        ("veh_h1_3", "route_h1", 110,  23.0),
        ("veh_h1_4", "route_h1", 160,  25.0),
        ("veh_h2_1", "route_h2",  20,  25.0),
        ("veh_h2_2", "route_h2",  70,  21.0),
        ("veh_h2_3", "route_h2", 120,  24.0),
        ("veh_h2_4", "route_h2", 170,  23.0),
        ("veh_h3_1", "route_h3",  15,  22.0),
        ("veh_h3_2", "route_h3",  60,  25.0),
        ("veh_h3_3", "route_h3", 105,  24.0),
        ("veh_h3_4", "route_h3", 150,  23.0),
        ("veh_h4_1", "route_h4",  20,  24.0),
        ("veh_h4_2", "route_h4",  70,  25.0),
        ("veh_h4_3", "route_h4", 120,  21.0),
        ("veh_h4_4", "route_h4", 170,  26.0),
        ("veh_v1_1", "route_v1",  20,  20.0),
        ("veh_v1_2", "route_v1",  80,  19.0),
        ("veh_v1_3", "route_v1", 140,  18.0),
        ("veh_v2_1", "route_v2",  20,  19.0),
        ("veh_v2_2", "route_v2",  90,  20.0),
        ("veh_v2_3", "route_v2", 160,  19.0),
        ("veh_v3_1", "route_v3",  25,  18.0),
        ("veh_v3_2", "route_v3",  95,  20.0),
        ("veh_v3_3", "route_v3", 165,  19.0),
        ("veh_v4_1", "route_v4",  25,  20.0),
        ("veh_v4_2", "route_v4",  95,  18.0),
        ("veh_v4_3", "route_v4", 160,  21.0),
    ]
    # Track how many times each vehicle has been respawned
    respawn_counts = {}
    respawn_counter = 0   # used to make unique ids on respawn

    try:
        while step < SIM_STEPS:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            vehicles = list(traci.vehicle.getIDList())

            # ── Respawn vehicles that have finished their route ────────────
            active_ids = set(vehicles)
            for (orig_vid, route_id, dep_pos, dep_spd) in SPAWN_DEFS:
                count    = respawn_counts.get(orig_vid, 0)
                cur_id   = orig_vid if count == 0 else f"{orig_vid}_r{count}"
                if cur_id not in active_ids:
                    respawn_counter += 1
                    new_id = f"{orig_vid}_r{respawn_counts.get(orig_vid, 0) + 1}"
                    try:
                        traci.vehicle.add(
                            vehID       = new_id,
                            routeID     = route_id,
                            typeID      = "car",
                            depart      = "now",
                            departLane  = "best",
                            departPos   = str(dep_pos),
                            departSpeed = str(dep_spd),
                        )
                        respawn_counts[orig_vid] = respawn_counts.get(orig_vid, 0) + 1
                    except Exception:
                        pass

            # ── Congestion update ─────────────────────────────────────────
            congestion_analyzer.update(step, vehicles)

            # ── Dynamic routing ───────────────────────────────────────────
            routes_changed = 0
            for vid in vehicles:
                try:
                    current_edge = traci.vehicle.getRoadID(vid)
                    if current_edge.startswith(':'):
                        continue

                    if vid not in vehicle_goals:
                        all_goals = ['J2', 'J4', 'J6', 'J7', 'J8', 'J10', 'J12']
                        vehicle_goals[vid] = all_goals[len(vehicle_goals) % len(all_goals)]

                    goal = vehicle_goals[vid]

                    if route_manager.update_vehicle_route(vid, step, current_edge, cur, goal):
                        routes_changed += 1
                        new_route = route_manager.vehicle_routes[vid]
                        if new_route:
                            try:
                                traci.vehicle.setRoute(vid, new_route)
                            except Exception:
                                pass
                except Exception:
                    continue

            # ── Congestion DB log ─────────────────────────────────────────
            if step % 10 == 0:
                for edge, info in congestion_analyzer.congestion_levels.items():
                    try:
                        cur.execute(
                            "INSERT INTO congestion_log VALUES (?, ?, ?, ?, ?, ?)",
                            (step, sim_time, edge,
                             info['count'], info['avg_speed'], info['level']))
                    except Exception:
                        pass

            # ── DQN step + speed control + brake visuals ─────────────────
            warnings = {}
            for vid in vehicles:
                current_state             = agent.get_state(vid, vehicles)
                action                    = agent.act(current_state)
                action_name, speed_factor = agent.actions[action]

                try:
                    current_speed = traci.vehicle.getSpeed(vid)
                    pos           = traci.vehicle.getPosition(vid)
                    is_too_close  = False
                    min_dist      = float('inf')

                    for other in vehicles:
                        if other == vid:
                            continue
                        try:
                            opos = traci.vehicle.getPosition(other)
                            d    = distance(pos, opos)
                            if d < min_dist:
                                min_dist = d
                            if d < COLOR_CHANGE_THRESHOLD:
                                is_too_close = True
                        except Exception:
                            continue

                    traci.vehicle.setSpeedMode(vid, 31)
                    traci.vehicle.setLaneChangeMode(vid, 0)
                    traci.vehicle.setMaxSpeed(vid, MAX_SPEED)
                    traci.vehicle.setMinGap(vid, 5.0)
                    traci.vehicle.setTau(vid, 1.5)
                    traci.vehicle.setDecel(vid, 6.0)
                    traci.vehicle.setEmergencyDecel(vid, 9.0)

                    # ── Distance-based speed control ───────────────────────
                    if min_dist <= STOP_DIST:
                        target_speed = 2.0

                    elif min_dist <= SAFE_DISTANCE:
                        ratio        = (min_dist - STOP_DIST) / (SAFE_DISTANCE - STOP_DIST)
                        target_speed = 2.0 + ratio * 3.0

                    elif min_dist <= BRAKE_HARD_DIST:
                        ratio        = (min_dist - SAFE_DISTANCE) / (BRAKE_HARD_DIST - SAFE_DISTANCE)
                        target_speed = 5.0 + ratio * 4.0

                    elif min_dist <= BRAKE_EARLY_DIST:
                        target_speed = NORMAL_SPEED * 0.60

                    else:
                        target_speed = NORMAL_SPEED

                    target_speed = min(target_speed, MAX_SPEED)

                    speed_diff   = target_speed - current_speed
                    if speed_diff > 0:
                        smooth_speed = current_speed + min(2.0, speed_diff)
                    else:
                        smooth_speed = current_speed + max(-4.0, speed_diff)
                    smooth_speed = max(2.0, min(smooth_speed, MAX_SPEED))

                    traci.vehicle.setSpeed(vid, smooth_speed)

                    # ── Determine action name for display ─────────────────
                    if min_dist <= STOP_DIST:
                        action_name = "STOP"
                        action      = 3
                    elif min_dist <= SAFE_DISTANCE:
                        action_name = "hard_brake"
                        action      = 3
                    elif min_dist <= BRAKE_HARD_DIST:
                        action_name = "medium_brake"
                        action      = 2
                    elif min_dist <= BRAKE_EARLY_DIST:
                        action_name = "light_brake"
                        action      = 1

                    # ── Vehicle colour ────────────────────────────────────
                    if min_dist <= COLOR_CHANGE_THRESHOLD:
                        traci.vehicle.setColor(vid, (255, 0, 0, 255))
                        warnings[vid] = [f"CLOSE! ({min_dist:.1f}m) {smooth_speed:.1f}m/s"]
                    elif min_dist <= BRAKE_EARLY_DIST:
                        traci.vehicle.setColor(vid, (255, 140, 0, 255))
                        warnings[vid] = [f"BRAKE ({min_dist:.1f}m) {smooth_speed:.1f}m/s"]
                    else:
                        traci.vehicle.setColor(vid, (0, 200, 0, 255))
                        edge_info = congestion_analyzer.get_edge_info(
                            traci.vehicle.getRoadID(vid))
                        warnings[vid] = [
                            f"Free ({min_dist:.0f}m) {smooth_speed:.1f}m/s "
                            f"| {edge_info['level']}"]

                    # ── Brake POI visual indicator ─────────────────────────
                    brake_indicators.update(vid, action, pos)

                    # ── FIX: Print vehicle line for Live Vehicle Monitor ───
                    # Format: VEH: <id> | <edge> | <speed> | <status>
                    # Parsed by index.html parseLine() to update the table
                    veh_status = (
                        "CLOSE"  if min_dist <= COLOR_CHANGE_THRESHOLD else
                        "BRAKE"  if min_dist <= BRAKE_EARLY_DIST else
                        "Free"
                    )
                    print(
                        f"VEH: {vid} | {traci.vehicle.getRoadID(vid)} | "
                        f"{smooth_speed:.1f} | {veh_status}",
                        flush=True)

                except Exception:
                    pass

                # ── Experience replay memory ──────────────────────────────
                reward = agent.calculate_reward(vid, vehicles, action)
                if vid in vehicle_states:
                    prev_state, prev_action = vehicle_states[vid]
                    done = vid not in vehicles
                    agent.remember(prev_state, prev_action, reward, current_state, done)
                vehicle_states[vid] = (current_state, action)
                episode_rewards.append(reward)

            # Remove brake indicators for vehicles that left the simulation
            active_set = set(vehicles)
            for vid in list(brake_indicators._active.keys()):
                if vid not in active_set:
                    brake_indicators._remove(vid)

            # ── Train every 10 steps ──────────────────────────────────────
            if step % 15 == 0:
                agent.replay()

            # ── Vehicle DB log ────────────────────────────────────────────
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

            # ── Console progress — printed EVERY step for live counter ──
            avg_reward = float(np.mean(episode_rewards[-100:])) if episode_rewards else 0
            cr = congestion_analyzer.get_congestion_report()
            print(
                f"Step {step:4d} | Time {sim_time:6.1f}s | "
                f"Vehicles: {len(vehicles):2d} | "
                f"e: {agent.epsilon:.3f} | Reward: {avg_reward:+6.2f}",
                flush=True)
            if step % 50 == 0:
                print(
                    f"    Congestion: HIGH={len(cr['HIGH'])} "
                    f"MED={len(cr['MEDIUM'])} "
                    f"LOW={len(cr['LOW'])} | "
                    f"Routes changed: {routes_changed}",
                    flush=True)

            step += 1

    finally:
        print("\n" + "=" * 70)
        print("Closing SUMO and saving model...")

        brake_indicators.remove_all()

        agent.save_model(model_path)

        try:
            import pickle
            stats_path = os.path.join(SCRIPT_DIR, "training_stats.pkl")
            with open(stats_path, 'wb') as f:
                pickle.dump({
                    'episode_rewards': episode_rewards,
                    'final_epsilon':   agent.epsilon,
                    'training_steps':  agent.training_step,
                    'route_changes':   route_manager.route_changes,
                }, f)
            print("✓ Training stats saved")
        except Exception as e:
            print(f"WARNING: Could not save stats: {e}")

        try:
            traci.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

        print("=" * 70)
        print("Simulation finished!")
        print(f"  Total route changes : {len(route_manager.route_changes)}")
        print(f"  Database            : v2v_routing_logs.db")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()