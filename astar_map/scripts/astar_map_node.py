#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified A* path planner (OSM) with AUTO indoor/outdoor mode via ref.csv

Auto mode selection by CSV (2nd row):
- If abs(ref_lat)+abs(ref_lon) < 1e-8  --> INDOOR mode (LOCAL/ENU)
- Else                                 --> OUTDOOR mode (UTM-REL, origin = ref_east/ref_north)

Shared features:
  • Edge-orthogonal projection start snapping with forward hysteresis
  • Publish /astar/path only when changed (optional slow repub)
  • Optional Jump-Guard to blacklist abnormal long edges and replan
  • Auto-fallback: If UTM edges look insane (e.g., false-northing mix), rebuild in ENU

ROS Params (selected):
  ~osm_file (str)                 : OSM XML path
  ~ref_file (str)                 : CSV path (header includes ref_lat, ref_lon, ref_east, ref_north, ...)
  ~origin_yaw_deg (float)         : yaw-only rotation from local XY to map (deg). default 0.0
  ~jump_guard_enable (bool)       : default false
  ~jump_guard_max_step_m (float)  : default 20.0
  ~jump_guard_max_attempts (int)  : default 3
  ~debug_log_enable (bool)        : default true
  ~path_repub_period (float)      : default 0.0 (no periodic republish)
  ~snap_progress_min_step_m (float): default 3.0
  ~snap_back_allow_m (float)      : default 0.3
  ~mode_override (str)            : "AUTO"|"ENU"|"UTM". default "AUTO"

Topics:
  pub  /astar/graph_markers        (visualization_msgs/Marker)
  pub  /astar/path                 (nav_msgs/Path)
  pub  /astar/path_wgs84           (nav_msgs/Path; x=lat, y=lon just for debug)
  pub  /astar/path_node_id_list    (std_msgs/Int32MultiArray)
  pub  /astar/server_dst_node_list (sensor_msgs/PointCloud2)

  sub  /initialpose                                (geometry_msgs/PoseWithCovarianceStamped)
  sub  lio_localizer/odometry/optimization         (nav_msgs/Odometry)
  sub  /move_base_simple/goal                      (geometry_msgs/PoseStamped)
  sub  /server_to_robot_topic                      (astar_map/server_to_robot)
"""

import rospy, math, sys, time, csv, colorsys, struct, xml.etree.ElementTree as ET
from geometry_msgs.msg import Point, PoseStamped, Quaternion, PoseWithCovarianceStamped
from nav_msgs.msg import Path, Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Header, Int32MultiArray
from sensor_msgs.msg import PointCloud2, PointField
from astar_map.msg import server_to_robot
import utm

# -------------------- Data --------------------
class Node:
    def __init__(self, id_, east, north, lat, lon, name=None):
        self.id = id_
        self.east = east    # local XY before yaw (ENU-relative or UTM-relative)
        self.north = north
        self.lat = lat
        self.lon = lon
        self.name = name
        self.parent = None
        self.cost = sys.maxsize
    def __eq__(self, o): return self.id == o.id

class Edge:
    def __init__(self, src, dst):
        self.src = src; self.dst = dst

# -------------------- Planner --------------------
class AStarPlanner:
    def __init__(self):
        # Graph
        self.node_list = []
        self.edge_list = []

        # Mode
        self.mode = "AUTO"  # AUTO | ENU | UTM
        self._mode_param = rospy.get_param("~mode_override", "AUTO").upper()
        self._mode_locked = False

        # Yaw (local->map)
        self._yaw_deg = float(rospy.get_param("~origin_yaw_deg", 0.0))
        self._cos = math.cos(math.radians(self._yaw_deg))
        self._sin = math.sin(math.radians(self._yaw_deg))

        # ENU reference (when mode ENU)
        self._ref_lat = None
        self._ref_lon = None
        self._ecef_ref = None
        self._enu_R = None  # 3x3

        # UTM reference from CSV (when mode UTM)
        self._utm_ref_e = None
        self._utm_ref_n = None

        # Start/Goal
        self.start_id = None
        self.goal_id = None
        self.start_init_flag = False
        self.new_goal_flag = False
        self.server_dst_node_list = []

        # Jump-guard & debug
        self.jump_guard_enable = rospy.get_param("~jump_guard_enable", False)
        self.jump_guard_max_step_m = float(rospy.get_param("~jump_guard_max_step_m", 20.0))
        self.jump_guard_max_attempts = int(rospy.get_param("~jump_guard_max_attempts", 3))
        self.bad_edges = set()
        self.debug_log_enable = rospy.get_param("~debug_log_enable", True)
        self.path_repub_period = float(rospy.get_param("~path_repub_period", 0.0))
        self._last_path_nodes = None
        self._last_path_pub_t = 0.0

        # Snapping state
        self._snap_edge = None
        self._snap_t = None
        self._snap_last_update_s = 0.0
        self._snap_progress_min_step_m = float(rospy.get_param("~snap_progress_min_step_m", 3.0))
        self._snap_back_allow_m = float(rospy.get_param("~snap_back_allow_m", 0.3))

        # Pubs/Subs
        self.pub_marker = rospy.Publisher('/astar/graph_markers', Marker, queue_size=10)
        self.pub_path = rospy.Publisher('/astar/path', Path, queue_size=10)
        self.pub_path_wgs84 = rospy.Publisher('/astar/path_wgs84', Path, queue_size=10)
        self.pub_path_node_id_list = rospy.Publisher('/astar/path_node_id_list', Int32MultiArray, queue_size=10)
        self.pub_server_dst_list = rospy.Publisher('/astar/server_dst_node_list', PointCloud2, queue_size=10)

        self.sub_start_from_rviz = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.callback_start)
        self.sub_start_from_pose = rospy.Subscriber('lio_localizer/odometry/optimization', Odometry, self.pose_callback)
        self.sub_goal_from_rviz = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.callback_goal_from_rviz)
        self.sub_goal_from_server = rospy.Subscriber('/server_to_robot_topic', server_to_robot, self.callback_goal_from_server)

    # ---------- ENU helpers ----------
    _A = 6378137.0
    _E2 = 6.69437999014e-3

    @staticmethod
    def _llh_to_ecef(lat_deg, lon_deg, h=0.0):
        lat = math.radians(lat_deg); lon = math.radians(lon_deg)
        a = AStarPlanner._A; e2 = AStarPlanner._E2
        sin_lat = math.sin(lat); cos_lat = math.cos(lat)
        sin_lon = math.sin(lon); cos_lon = math.cos(lon)
        N = a / math.sqrt(1.0 - e2 * sin_lat*sin_lat)
        x = (N + h) * cos_lat * cos_lon
        y = (N + h) * cos_lat * sin_lon
        z = (N * (1.0 - e2) + h) * sin_lat
        return x, y, z

    @staticmethod
    def _ecef_to_enu_matrix(ref_lat_deg, ref_lon_deg):
        lat = math.radians(ref_lat_deg); lon = math.radians(ref_lon_deg)
        sin_lat = math.sin(lat); cos_lat = math.cos(lat)
        sin_lon = math.sin(lon); cos_lon = math.cos(lon)
        R = [
            [-sin_lon,              cos_lon,               0.0],
            [-sin_lat*cos_lon,     -sin_lat*sin_lon,      cos_lat],
            [ cos_lat*cos_lon,      cos_lat*sin_lon,      sin_lat],
        ]
        return R

    @staticmethod
    def _mat3_mul_vec3(M, v):
        return (
            M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
            M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
            M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
        )

    def _ll_to_enu(self, lat_deg, lon_deg):
        x, y, z = self._llh_to_ecef(lat_deg, lon_deg, 0.0)
        dx, dy, dz = x - self._ecef_ref[0], y - self._ecef_ref[1], z - self._ecef_ref[2]
        e, n, _ = self._mat3_mul_vec3(self._enu_R, (dx, dy, dz))
        return e, n

    # ---------- local -> map (yaw only) ----------
    def _xy_to_map(self, x, y):
        mx =  self._cos * x + self._sin * y
        my = -self._sin * x + self._cos * y
        return mx, my

    # ---------- CSV reader & mode decide ----------
    @staticmethod
    def _read_ref_csv(path):
        try:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                row = next(reader, None)
                if header is None or row is None:
                    return None
                h2i = {h.strip(): i for i, h in enumerate(header)}
                def get(name, default=0.0):
                    i = h2i.get(name, None)
                    if i is None or i >= len(row): return default
                    try: return float(row[i])
                    except: return default
                ref = {
                    "ref_lat":   get("ref_lat", 0.0),
                    "ref_lon":   get("ref_lon", 0.0),
                    "ref_east":  get("ref_east", 0.0),
                    "ref_north": get("ref_north", 0.0),
                }
                return ref
        except Exception as e:
            rospy.logwarn(f"[astar] failed to read ref CSV: {e}")
            return None

    def _decide_mode_from_csv(self, refcsv):
        if self._mode_param in ("ENU", "UTM"):
            self.mode = "ENU" if self._mode_param == "ENU" else "UTM"
            self._mode_locked = True
            rospy.loginfo(f"[astar] mode_override='{self._mode_param}'")
            return

        if refcsv is None:
            self.mode = "ENU"
            rospy.logwarn("[astar] no CSV → default ENU (indoor-safe).")
            return

        lat = abs(refcsv.get("ref_lat", 0.0))
        lon = abs(refcsv.get("ref_lon", 0.0))
        if lat + lon < 1e-8:
            self.mode = "ENU"
            rospy.loginfo("[astar] CSV lat/lon == 0 → INDOOR → ENU mode.")
        else:
            self.mode = "UTM"
            rospy.loginfo("[astar] CSV lat/lon present → OUTDOOR → UTM-REL mode.")

    # -------------------- Loading --------------------
    def load_osm_data(self, osm_file, ref_file):
        root = ET.parse(osm_file).getroot()
        first_nd = root.find('.//node')
        if first_nd is None:
            rospy.logerr("[astar] OSM has no nodes."); return

        # CSV & mode
        refcsv = self._read_ref_csv(ref_file) if ref_file else None
        self._decide_mode_from_csv(refcsv)

        # ENU reference (for ENU OR for fallback)
        f_id = int(first_nd.attrib['id'])
        f_lat = float(first_nd.attrib['lat']); f_lon = float(first_nd.attrib['lon'])

        if self.mode == "ENU":
            self._ref_lat = f_lat if refcsv is None or abs(refcsv.get("ref_lat", 0.0))+abs(refcsv.get("ref_lon", 0.0))<1e-8 else refcsv["ref_lat"]
            self._ref_lon = f_lon if refcsv is None or abs(refcsv.get("ref_lat", 0.0))+abs(refcsv.get("ref_lon", 0.0))<1e-8 else refcsv["ref_lon"]
            rospy.loginfo(f"[astar] ENU origin lat={self._ref_lat:.7f}, lon={self._ref_lon:.7f}")
            self._ecef_ref = self._llh_to_ecef(self._ref_lat, self._ref_lon, 0.0)
            self._enu_R = self._ecef_to_enu_matrix(self._ref_lat, self._ref_lon)

            self.node_list = []
            for nd in root.findall('.//node'):
                nid = int(nd.attrib['id'])
                lat = float(nd.attrib['lat']); lon = float(nd.attrib['lon'])
                tag = nd.find('tag[@k="name"]')
                name = tag.attrib['v'] if tag is not None else None
                e, n = self._ll_to_enu(lat, lon)
                self.node_list.append(Node(nid, e, n, lat, lon, name))

        else:  # UTM-REL
            self._utm_ref_e = refcsv.get("ref_east", 0.0) if refcsv else 0.0
            self._utm_ref_n = refcsv.get("ref_north", 0.0) if refcsv else 0.0
            rospy.loginfo(f"[astar] UTM-REL origin east={self._utm_ref_e:.3f}, north={self._utm_ref_n:.3f}")

            # Also keep ENU reference ready for possible fallback
            self._ref_lat = f_lat; self._ref_lon = f_lon
            self._ecef_ref = self._llh_to_ecef(self._ref_lat, self._ref_lon, 0.0)
            self._enu_R = self._ecef_to_enu_matrix(self._ref_lat, self._ref_lon)

            self.node_list = []
            big_jump = False
            for nd in root.findall('.//node'):
                nid = int(nd.attrib['id'])
                lat = float(nd.attrib['lat']); lon = float(nd.attrib['lon'])
                tag = nd.find('tag[@k="name"]')
                name = tag.attrib['v'] if tag is not None else None
                ue, un, znum, zlet = utm.from_latlon(lat, lon)
                x = ue - self._utm_ref_e
                y = un - self._utm_ref_n
                self.node_list.append(Node(nid, x, y, lat, lon, name))

            # quick sanity on edges (false-northing mix → 10,000 km jumps)
            mx = 0.0; cnt = 0
            for way in root.findall('.//way'):
                ids = way.findall('nd')
                for i in range(len(ids) - 1):
                    a = self.findNodeById(int(ids[i].attrib['ref']))
                    b = self.findNodeById(int(ids[i+1].attrib['ref']))
                    if a and b:
                        d = math.hypot(a.east - b.east, a.north - b.north)
                        mx = max(mx, d); cnt += 1
            if cnt > 0 and mx > 5_000.0 and not self._mode_locked:
                rospy.logwarn(f"[astar] abnormal UTM edge length max={mx:.1f} m → fallback to ENU.")
                # rebuild nodes in ENU
                self.mode = "ENU"
                self.node_list = []
                self._ecef_ref = self._llh_to_ecef(self._ref_lat, self._ref_lon, 0.0)
                self._enu_R = self._ecef_to_enu_matrix(self._ref_lat, self._ref_lon)
                for nd in root.findall('.//node'):
                    nid = int(nd.attrib['id'])
                    lat = float(nd.attrib['lat']); lon = float(nd.attrib['lon'])
                    tag = nd.find('tag[@k="name"]')
                    name = tag.attrib['v'] if tag is not None else None
                    e, n = self._ll_to_enu(lat, lon)
                    self.node_list.append(Node(nid, e, n, lat, lon, name))

        # Build edges (bidirectional)
        self.edge_list = []
        lens = []
        for way in root.findall('.//way'):
            ids = way.findall('nd')
            for i in range(len(ids) - 1):
                a = self.findNodeById(int(ids[i].attrib['ref']))
                b = self.findNodeById(int(ids[i+1].attrib['ref']))
                if a is None or b is None: continue
                d = math.hypot(a.east - b.east, a.north - b.north)
                lens.append(d)
                self.edge_list.append(Edge(a, b))
                self.edge_list.append(Edge(b, a))

        if lens:
            lens.sort()
            med = lens[len(lens)//2]; p90 = lens[int(0.9*len(lens))]; mx = lens[-1]
            rospy.loginfo(f"[astar] edges built ({self.mode}) len median={med:.2f} p90={p90:.2f} max={mx:.2f} m")

    # -------------------- Graph helpers --------------------
    def set_dst_node_list(self, lst): self.server_dst_node_list = lst

    def graph_setup(self):
        for n in self.node_list:
            n.parent = None; n.cost = sys.maxsize

    def findNodeById(self, id_):
        for n in self.node_list:
            if n.id == id_: return n
        return None

    # -------------------- A* --------------------
    def edges(self, node):
        return [e for e in self.edge_list if e.src == node and (e.src.id, e.dst.id) not in self.bad_edges]

    def distance(self, a, b):
        return math.hypot(a.east - b.east, a.north - b.north)

    def planning(self, start_id, goal_id):
        s = self.findNodeById(start_id); g = self.findNodeById(goal_id)
        if s is None or g is None: return []
        open_set = {s.id: s}; closed = {}
        s.cost = 0.0
        while open_set:
            c_id = min(open_set, key=lambda i: open_set[i].cost + self.distance(g, open_set[i]))
            cur = open_set[c_id]
            if cur == g: break
            del open_set[c_id]; closed[c_id] = cur
            for e in self.edges(cur):
                nid = e.dst.id
                if nid in closed: continue
                new_cost = e.src.cost + self.distance(e.src, e.dst)
                if e.dst.cost > new_cost:
                    e.dst.cost = new_cost; e.dst.parent = e.src
                if nid not in open_set: open_set[nid] = e.dst
        return self._backtrack(g)

    def _backtrack(self, node):
        path = []; cur = node
        while cur and cur.parent is not None:
            path.append(cur.id); cur = cur.parent
        if cur: path.append(cur.id)
        path.reverse(); return path

    # -------------------- Jump Guard --------------------
    def _seg_len_map(self, u, v):
        ux, uy = self._xy_to_map(u.east, u.north)
        vx, vy = self._xy_to_map(v.east, v.north)
        return math.hypot(vx - ux, vy - uy)

    def validate_or_blacklist(self, path_ids):
        if not self.jump_guard_enable or len(path_ids) < 2: return path_ids
        thr = self.jump_guard_max_step_m
        for u_id, v_id in zip(path_ids, path_ids[1:]):
            u = self.findNodeById(u_id); v = self.findNodeById(v_id)
            if u is None or v is None: continue
            if self._seg_len_map(u, v) > thr:
                rospy.logwarn(f"[jump_guard] abnormal jump {u_id}->{v_id} (> {thr} m). Blacklist & replan.")
                self.bad_edges.add((u_id, v_id)); self.bad_edges.add((v_id, u_id))
                return None
        return path_ids

    # -------------------- Edge-projection snapping --------------------
    def _nearest_projection_on_edges(self, x, y):
        best_e = None; best_t = 0.0; best_px = best_py = 0.0; best_d2 = 1e18; best_len = 0.0
        for e in self.edge_list:
            sx, sy = self._xy_to_map(e.src.east, e.src.north)
            dx, dy = self._xy_to_map(e.dst.east, e.dst.north)
            vx, vy = dx - sx, dy - sy
            denom = vx*vx + vy*vy
            if denom < 1e-12:
                t = 0.0; px, py = sx, sy; elen = 0.0
            else:
                t = ((x - sx)*vx + (y - sy)*vy) / denom
                t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                px, py = sx + t*vx, sy + t*vy
                elen = math.sqrt(denom)
            d2 = (x - px)**2 + (y - py)**2
            if d2 < best_d2:
                best_d2 = d2; best_e = e; best_t = t; best_px = px; best_py = py; best_len = elen
        return best_e, best_t, best_px, best_py, best_d2, best_len

    def _accept_or_clamp_projection(self, edge, t, edge_len):
        progressed_m = 0.0
        if self._snap_edge is None:
            self._snap_edge = (edge.src.id, edge.dst.id); self._snap_t = float(t)
            return self._snap_edge, self._snap_t, progressed_m

        prev_src, prev_dst = self._snap_edge
        cur_pair = (edge.src.id, edge.dst.id)

        if cur_pair == self._snap_edge:
            if t + (self._snap_back_allow_m / max(edge_len, 1e-6)) >= self._snap_t:
                progressed_m = (t - self._snap_t) * edge_len
                self._snap_t = max(self._snap_t, t)
            return self._snap_edge, self._snap_t, max(progressed_m, 0.0)

        if prev_dst == edge.src.id and (self._snap_t is not None and self._snap_t > 0.9):
            self._snap_edge = cur_pair; self._snap_t = float(t)
            progressed_m = t * edge_len
            return self._snap_edge, self._snap_t, progressed_m

        return self._snap_edge, self._snap_t, 0.0

    # -------------------- Visualization --------------------
    def _smooth_path(self, path):
        """Remove redundant waypoints and smooth path (remove backtracking)."""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        i = 1
        while i < len(path):
            # Skip duplicate nodes
            if path[i] == smoothed[-1]:
                i += 1
                continue
            
            # Skip nodes that cause backtracking (simple check)
            if len(smoothed) >= 2:
                n_prev = self.findNodeById(smoothed[-2])
                n_curr = self.findNodeById(smoothed[-1])
                n_next = self.findNodeById(path[i])
                
                if n_prev and n_curr and n_next:
                    # Calculate distances
                    d_prev_curr = self.distance(n_prev, n_curr)
                    d_curr_next = self.distance(n_curr, n_next)
                    d_prev_next = self.distance(n_prev, n_next)
                    
                    # If direct path is significantly shorter, skip middle node
                    if d_prev_next < 0.9 * (d_prev_curr + d_curr_next):
                        smoothed.pop()  # Remove last node
                        continue
            
            smoothed.append(path[i])
            i += 1
        
        return smoothed

    def show_path(self, path, stamp=None):
        if not path: return
        if stamp is None: stamp = rospy.Time.now()
        
        # Smooth path to remove redundant waypoints
        path = self._smooth_path(path)
        
        p = Path(); p.header.frame_id = "map"; p.header.stamp = stamp
        pw = Path(); pw.header.frame_id = "map"; pw.header.stamp = stamp

        for nid in path[:-1]:
            n = self.findNodeById(nid)
            if n is None: continue
            x, y = self._xy_to_map(n.east, n.north)
            ps = PoseStamped(); ps.header = p.header
            ps.pose.position.x = x; ps.pose.position.y = y; ps.pose.position.z = 0.0
            p.poses.append(ps)

            pwps = PoseStamped(); pwps.header = pw.header
            pwps.pose.position.x = n.lat; pwps.pose.position.y = n.lon; pwps.pose.position.z = 0.0
            pw.poses.append(pwps)

        last = self.findNodeById(path[-1])
        if last:
            x, y = self._xy_to_map(last.east, last.north)
            ps = PoseStamped(); ps.header = p.header
            ps.pose.position.x = x; ps.pose.position.y = y; ps.pose.position.z = 0.0
            p.poses.append(ps)

            pwps = PoseStamped(); pwps.header = pw.header
            pwps.pose.position.x = last.lat; pwps.pose.position.y = last.lon; pwps.pose.position.z = 0.0
            pw.poses.append(pwps)

        self.pub_path.publish(p)
        self.pub_path_wgs84.publish(pw)

    def show_server_dst_nodes(self):
        npt = len(self.server_dst_node_list)
        if npt == 0: return
        hues = [0.67] if npt == 1 else [(0.67 - i/float(npt)) % 1.0 for i in range(npt)]
        colors = [struct.unpack('I', struct.pack('BBBB',
                   int(colorsys.hsv_to_rgb(h,1.0,1.0)[0]*255),
                   int(colorsys.hsv_to_rgb(h,1.0,1.0)[1]*255),
                   int(colorsys.hsv_to_rgb(h,1.0,1.0)[2]*255),
                   255))[0] for h in hues]
        fields = [PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                  PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                  PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                  PointField(name="rgba", offset=12, datatype=PointField.UINT32, count=1)]
        buf = []
        for nid, color in zip(self.server_dst_node_list, colors):
            n = self.findNodeById(nid)
            if n is None: continue
            x, y = self._xy_to_map(n.east, n.north)
            buf.append(struct.pack('fffI', float(x), float(y), 0.0, int(color)))
        header = Header(frame_id="map", stamp=rospy.Time.now())
        cloud = PointCloud2(header=header, height=1, width=len(buf), is_dense=True, is_bigendian=False,
                            fields=fields, point_step=16, row_step=16*len(buf), data=b''.join(buf))
        self.pub_server_dst_list.publish(cloud)

    def visualize_graph(self):
        m = Marker(); m.header.frame_id = "map"; m.header.stamp = rospy.Time.now()
        m.type = Marker.LINE_LIST; m.action = Marker.ADD; m.scale.x = 0.3
        m.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        white = ColorRGBA(1,1,1,1)
        for e in self.edge_list:
            sx, sy = self._xy_to_map(e.src.east, e.src.north)
            dx, dy = self._xy_to_map(e.dst.east, e.dst.north)
            m.points.append(Point(sx, sy, -0.1)); m.points.append(Point(dx, dy, -0.1))
            m.colors.append(white);               m.colors.append(white)
        self.pub_marker.publish(m)

    # -------------------- Callbacks --------------------
    def _find_nearest_node_simple(self, x, y):
        best=None; best_d2=1e18
        for n in self.node_list:
            nx, ny = self._xy_to_map(n.east, n.north)
            d2 = (nx-x)**2 + (ny-y)**2
            if d2 < best_d2: best_d2=d2; best=n
        return best

    def _snap_and_update_from_position(self, position):
        x, y = position.x, position.y
        edge, t, _, _, _, elen = self._nearest_projection_on_edges(x, y)
        if edge is None:
            return self._find_nearest_node_simple(x, y)

        (acc_edge, acc_t, progressed_m) = self._accept_or_clamp_projection(edge, t, elen)
        self._snap_last_update_s += progressed_m

        src_id, dst_id = acc_edge
        src = self.findNodeById(src_id); dst = self.findNodeById(dst_id)
        chosen = dst if acc_t >= 0.5 else src

        if self._snap_last_update_s >= self._snap_progress_min_step_m:
            self._snap_last_update_s = 0.0
            return chosen
        return self.findNodeById(self.start_id) if self.start_id is not None else chosen

    def callback_start(self, data):
        n = self._snap_and_update_from_position(data.pose.pose.position)
        if n:
            self.start_id = n.id; self.start_init_flag = True
            if self.debug_log_enable:
                rospy.loginfo(f"[astar] start set by /initialpose → node {n.id}")

    def pose_callback(self, data):
        n = self._snap_and_update_from_position(data.pose.pose.position)
        if n:
            self.start_id = n.id; self.start_init_flag = True

    def callback_goal_from_rviz(self, data):
        n = self._find_nearest_node_simple(data.pose.position.x, data.pose.position.y)
        if n and self.goal_id != n.id:
            self.goal_id = n.id; self.new_goal_flag = True
            if self.debug_log_enable:
                rospy.loginfo(f"[astar] goal set by RViz → node {n.id}")

    def callback_goal_from_server(self, data):
        if self.server_dst_node_list and 0 <= data.Cmd_dest_index < len(self.server_dst_node_list):
            nid = self.server_dst_node_list[data.Cmd_dest_index]
            n = self.findNodeById(nid)
            if n and self.goal_id != n.id:
                self.goal_id = n.id; self.new_goal_flag = True
                if self.debug_log_enable:
                    rospy.loginfo(f"[astar] goal set by server index → node {n.id}")
        elif data.Cmd_dest_lat > 0.01 and data.Cmd_dest_lon > 0.01:
            # WGS84 dest → local map XY (mode-aware)
            if self.mode == "UTM":
                ue, un, _, _ = utm.from_latlon(data.Cmd_dest_lat, data.Cmd_dest_lon)
                mx, my = self._xy_to_map(ue - self._utm_ref_e, un - self._utm_ref_n)
            else:
                e, n = self._ll_to_enu(data.Cmd_dest_lat, data.Cmd_dest_lon)
                mx, my = self._xy_to_map(e, n)
            p = Point(); p.x = mx; p.y = my
            g = self._find_nearest_node_simple(p.x, p.y)
            if g and self.goal_id != g.id:
                self.goal_id = g.id; self.new_goal_flag = True
                if self.debug_log_enable:
                    rospy.loginfo(f"[astar] goal set by server WGS84 → node {g.id}")

    # -------------------- Path publish control --------------------
    def publish_path_if_changed(self, path_nodes):
        now = rospy.Time.now()
        changed = (self._last_path_nodes != path_nodes)
        do_periodic = False
        if not changed and self.path_repub_period > 0.0:
            tnow = time.monotonic()
            if (tnow - self._last_path_pub_t) >= self.path_repub_period:
                do_periodic = True; self._last_path_pub_t = tnow

        if changed:
            self._last_path_nodes = list(path_nodes)
            self._last_path_pub_t = time.monotonic()
            msg = Int32MultiArray(); msg.data = path_nodes
            self.pub_path_node_id_list.publish(msg)
            self.show_path(path_nodes, stamp=now)
            if self.debug_log_enable:
                rospy.loginfo(f"[astar] path published ({len(path_nodes)} nodes) [mode={self.mode}]")
        elif do_periodic:
            self.show_path(path_nodes, stamp=now)
            if self.debug_log_enable:
                rospy.loginfo("[astar] path republished (periodic)")

# -------------------- Main --------------------
if __name__ == "__main__":
    try:
        rospy.init_node('astar_map_node')
        _ = rospy.Publisher('/path', Path, queue_size=10)  # legacy compat

        a = AStarPlanner()
        osm = rospy.get_param("astar_map_node/osm_file")
        ref = rospy.get_param("astar_map_node/ref_file", "")
        a.load_osm_data(osm, ref)

        dst_str = rospy.get_param('~server_dst_node_list', "")
        if dst_str:
            a.set_dst_node_list([int(s) for s in dst_str.split(',') if s.strip()])

        rate = rospy.Rate(2)
        path_nodes = []

        while not rospy.is_shutdown():
            a.visualize_graph()
            a.show_server_dst_nodes()

            if a.start_init_flag and a.new_goal_flag:
                a.graph_setup()
                path_nodes = a.planning(a.start_id, a.goal_id)

                attempt = 0
                while a.jump_guard_enable and path_nodes and attempt < a.jump_guard_max_attempts:
                    validated = a.validate_or_blacklist(path_nodes)
                    if validated is not None:
                        path_nodes = validated; break
                    a.graph_setup(); path_nodes = a.planning(a.start_id, a.goal_id); attempt += 1

                a.new_goal_flag = False
                if path_nodes:
                    a.publish_path_if_changed(path_nodes)
                else:
                    rospy.logwarn("[astar] path not found")

            if path_nodes and a.path_repub_period > 0.0:
                a.publish_path_if_changed(path_nodes)

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
