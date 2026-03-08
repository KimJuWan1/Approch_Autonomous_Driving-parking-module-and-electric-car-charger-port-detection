#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera-based parking slot detector (BEV + Hough, robust parallel-pairing).
- ROI 교차 허용, 라인 병합, 페어링: 평행성 + 폭(후보/구간) + 겹침길이
- 각도 0/90 고정 필터 비사용(옵션), 자동으로 임의 각도에서도 동작
- publish 상한/FPS 제한, 단계별 디버그 + 거리 샘플 로그

출력:
- /parking/slots (auto_parking/ParkingSlotArray)
- /parking/best_slot (geometry_msgs/PoseStamped)
- /parking/debug/markers (visualization_msgs/MarkerArray)
"""

import rospy, numpy as np, cv2, math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from tf.transformations import quaternion_from_euler
from auto_parking.msg import ParkingSlot, ParkingSlotArray

# ---------- Utils ----------

def rot_to_quat(yaw: float) -> Quaternion:
    qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)
    return Quaternion(x=qx, y=qy, z=qz, w=qw)

def make_marker_box(frame_id, idx, cx, cy, yaw, w, l, z=0.02, color=(0.1,0.9,0.1,0.9)):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = rospy.Time.now()
    m.ns = "parking_slots"
    m.id = idx
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = cx; m.pose.position.y = cy; m.pose.position.z = z
    m.pose.orientation = rot_to_quat(yaw)
    m.scale.x = l; m.scale.y = w; m.scale.z = 0.02
    m.color = ColorRGBA(*color)
    m.lifetime = rospy.Duration(0.5)
    return m

def clamp(v, lo, hi): return lo if v < lo else (hi if v > hi else v)

def _seg_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    def ccw(x1,y1,x2,y2,x3,y3): return (y3-y1)*(x2-x1) - (y2-y1)*(x3-x1)
    d1 = ccw(ax,ay,bx,by,cx,cy); d2 = ccw(ax,ay,bx,by,dx,dy)
    d3 = ccw(cx,cy,dx,dy,ax,ay); d4 = ccw(cx,cy,dx,dy,bx,by)
    if (d1==0 and min(ax,bx)<=cx<=max(ax,bx) and min(ay,by)<=cy<=max(ay,by)): return True
    if (d2==0 and min(ax,bx)<=dx<=max(ax,bx) and min(ay,by)<=dy<=max(ay,by)): return True
    if (d3==0 and min(cx,dx)<=ax<=max(cx,dx) and min(cy,dy)<=ay<=max(cy,dy)): return True
    if (d4==0 and min(cx,dx)<=bx<=max(cx,dx) and min(cy,dy)<=by<=max(cy,dy)): return True
    return (d1>0) != (d2>0) and (d3>0) != (d4>0)

# ---------- Node ----------

class VisionParking:
    def __init__(self):
        self.bridge = CvBridge()

        # ===== 기본 파라미터 =====
        self.frame_id      = rospy.get_param("~frame_id", "map")
        self.base_frame    = rospy.get_param("~base_frame", "os_sensor")
        self.image_topic   = rospy.get_param("~image_topic", "/camera/image_raw")

        # BEV / 호모그래피
        self.m_per_px      = rospy.get_param("~bev_m_per_px", 0.01)
        self.bev_size_m    = rospy.get_param("~bev_size_m", [4.0, 3.0])
        H_list    = rospy.get_param("~bev_H", [])
        bev_src   = rospy.get_param("~bev_src_px", [])
        bev_dst_m = rospy.get_param("~bev_dst_m",  [])
        if len(H_list) == 9:
            self.H = np.array(H_list, dtype=np.float64).reshape(3,3)
            rospy.loginfo("[vision] using provided homography H.")
        elif len(bev_src)==4 and len(bev_dst_m)==4:
            src = np.array(bev_src, dtype=np.float32)
            dst = np.array(bev_dst_m, dtype=np.float32)
            self.H, _ = cv2.findHomography(src, dst)
            rospy.loginfo("[vision] computed H from 4-point correspondences.")
        else:
            rospy.logwarn("[vision] No homography provided. Using identity.")
            self.H = np.eye(3, dtype=np.float64)

        # 전방 제한(옵션)
        self.only_front  = rospy.get_param("~only_front", True)
        self.front_axis  = rospy.get_param("~front_axis", "x")
        self.front_min_m = rospy.get_param("~front_min_m", 0.30)

        # 이미지 처리
        self.gray_clip_limit = rospy.get_param("~gray_clip_limit", 3.0)
        self.adaptive        = rospy.get_param("~adaptive", True)
        self.binary_thr      = rospy.get_param("~binary_thr", 200)
        self.canny_low_high  = rospy.get_param("~canny_low_high", [50,150])

        # 허프 (픽셀 단위)
        self.hough_rho       = rospy.get_param("~hough_rho", 1.0)
        self.hough_theta_deg = rospy.get_param("~hough_theta_deg", 1.0)
        self.hough_thresh    = rospy.get_param("~hough_thresh", 45)
        self.hough_min_len   = rospy.get_param("~hough_min_len", 30)
        self.hough_max_gap   = rospy.get_param("~hough_max_gap", 5)

        # (옵션) 각도 필터(기본 OFF)
        self.angle_filter_enable = bool(rospy.get_param("~angle_filter_enable", False))
        self.angle_main_deg  = rospy.get_param("~angle_main_deg", [0.0, 90.0])
        self.angle_tol_deg   = rospy.get_param("~angle_tol_deg", 20.0)

        # 슬롯 규격/스코어
        self.slot_length     = rospy.get_param("~slot_length_m", 1.0)
        self.width_candidates= np.array(rospy.get_param("~width_candidates_m", [0.45,0.50,0.52]), dtype=np.float32)
        self.pair_width_tol  = rospy.get_param("~pair_width_tol_m", 0.06)

        # 폭 매칭 모드
        self.width_mode      = rospy.get_param("~width_mode", "range")  # "candidates" | "range"
        self.width_range_m   = rospy.get_param("~width_range_m", [0.30, 0.90])

        # 페어 평행성 허용
        self.pair_parallel_tol_deg = rospy.get_param("~pair_parallel_tol_deg", 12.0)

        self.center_step     = rospy.get_param("~center_step_m", 0.03)
        self.min_score       = rospy.get_param("~min_score", 0.20)

        # 점유(선택)
        self.use_cloud_for_occupancy = rospy.get_param("~use_cloud_for_occupancy", False)
        self.cloud_topic = rospy.get_param("~cloud_topic", "/ouster/points_map")
        self.occupied_ratio_thr = rospy.get_param("~occupied_ratio_thr", 0.06)
        self.obstacle_z_low  = rospy.get_param("~obstacle_z_low", 0.05)
        self.obstacle_z_high = rospy.get_param("~obstacle_z_high", 0.35)

        # ===== 혼잡 억제 & ROI =====
        self.use_roi = bool(rospy.get_param("~use_roi", True))
        self.autorelax_if_zero = bool(rospy.get_param("~autorelax_if_zero", True))
        self.roi_x_m = rospy.get_param("~roi_x_m", [0.3, 2.6])
        self.roi_y_m = rospy.get_param("~roi_y_m", [-0.9, 0.9])
        self.min_line_length_m    = rospy.get_param("~min_line_length_m", 0.18)  # ↓ 더 완화
        self.merge_angle_tol_deg  = rospy.get_param("~merge_angle_tol_deg", 4.0)
        self.merge_rho_tol_m      = rospy.get_param("~merge_rho_tol_m", 0.03)
        self.min_parallel_overlap_m = rospy.get_param("~min_parallel_overlap_m", 0.20)  # ↓ 더 완화
        self.max_lines_after_merge  = int(rospy.get_param("~max_lines_after_merge", 80))
        self.max_pairs_considered   = int(rospy.get_param("~max_pairs_considered", 300))
        self.max_slots_publish      = int(rospy.get_param("~max_slots_publish", 40))
        self.max_processing_fps     = float(rospy.get_param("~max_processing_fps", 8.0))
        self.publish_markers        = bool(rospy.get_param("~publish_markers", True))

        # pubs/subs
        self.pub_slots = rospy.Publisher("/parking/slots", ParkingSlotArray, queue_size=1)
        self.pub_best  = rospy.Publisher("/parking/best_slot", PoseStamped, queue_size=1)
        self.pub_mk    = rospy.Publisher("/parking/debug/markers", MarkerArray, queue_size=1)

        self.sub_img   = rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1, buff_size=2**22)
        self.last_cloud = None
        if self.use_cloud_for_occupancy and self.cloud_topic:
            self.sub_cloud = rospy.Subscriber(self.cloud_topic, PointCloud2, self.cb_cloud, queue_size=1)

        self._next_ok_time = 0.0
        self._dbg_counts = {}
        self._dbg_last_dists = []

        rospy.loginfo("[vision] subscribe image_topic=%s, occupancy=%s cloud=%s",
                      self.image_topic, str(self.use_cloud_for_occupancy), self.cloud_topic)

    # ---------- Helpers ----------

    def cb_cloud(self, msg): self.last_cloud = msg

    def img_to_ground(self, pts_uv):
        if len(pts_uv) == 0: return np.zeros((0,2), dtype=np.float32)
        uv1 = np.hstack([pts_uv, np.ones((len(pts_uv),1),dtype=np.float32)])
        XY1 = (self.H @ uv1.T).T
        XY  = XY1[:, :2] / np.clip(XY1[:,2:3], 1e-6, None)
        return XY.astype(np.float32)

    def _in_front_mask(self, XY):
        if not self.only_front: return np.ones((len(XY),), dtype=bool)
        return XY[:,0] >= self.front_min_m if self.front_axis.lower()=="x" else XY[:,1] >= self.front_min_m

    def _point_in_roi(self, x, y):
        x_min,x_max = self.roi_x_m; y_min,y_max = self.roi_y_m
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)

    def _seg_intersects_roi(self, x1,y1,x2,y2):
        x_min,x_max = self.roi_x_m; y_min,y_max = self.roi_y_m
        if self._point_in_roi(x1,y1) or self._point_in_roi(x2,y2): return True
        if (max(x1,x2) < x_min) or (min(x1,x2) > x_max) or (max(y1,y2) < y_min) or (min(y1,y2) > y_max): return False
        return (_seg_intersect(x1,y1,x2,y2, x_min,y_min, x_max,y_min) or
                _seg_intersect(x1,y1,x2,y2, x_max,y_min, x_max,y_max) or
                _seg_intersect(x1,y1,x2,y2, x_max,y_max, x_min,y_max) or
                _seg_intersect(x1,y1,x2,y2, x_min,y_max, x_min,y_min))

    # ---------- Lines from edges ----------
    def _lines_from_edges(self, edges, use_roi=True):
        lines = cv2.HoughLinesP(edges,
                                rho=self.hough_rho,
                                theta=np.deg2rad(self.hough_theta_deg),
                                threshold=self.hough_thresh,
                                minLineLength=self.hough_min_len,
                                maxLineGap=self.hough_max_gap)
        if lines is None:
            self._dbg_counts = dict(Hough=0, ROI_pass=0, Front_pass=0, Len_pass=0, Angle_pass=0, Accepted=0, use_roi=use_roi)
            return []

        lines = lines[:,0,:]
        n_hough = int(len(lines)); n_roi = n_front = n_len = n_ang = 0

        out=[]
        for x1,y1,x2,y2 in lines:
            uv = np.array([[x1,y1],[x2,y2]], dtype=np.float32)
            (x1g,y1g),(x2g,y2g) = self.img_to_ground(uv)

            if use_roi and not self._seg_intersects_roi(x1g,y1g,x2g,y2g): continue
            n_roi += 1
            if not (self._in_front_mask(np.array([[x1g,y1g]],dtype=np.float32))[0] or
                    self._in_front_mask(np.array([[x2g,y2g]],dtype=np.float32))[0]): continue
            n_front += 1

            dx, dy = (x2g-x1g), (y2g-y1g)
            length = math.hypot(dx, dy)
            if length < self.min_line_length_m: continue
            n_len += 1

            ang = math.atan2(dy, dx)
            if self.angle_filter_enable:
                ok=False
                for a0 in self.angle_main_deg:
                    if abs(((math.degrees(ang)-a0)+180)%360-180) <= self.angle_tol_deg:
                        ok=True; break
                if not ok: continue
                n_ang += 1
            else:
                n_ang = n_len  # 각도 필터 OFF

            nx, ny = -math.sin(ang), math.cos(ang)
            cx, cy = (x1g+x2g)/2.0, (y1g+y2g)/2.0

            out.append({
                "x1":x1g,"y1":y1g,"x2":x2g,"y2":y2g,
                "ang":ang,"nx":nx,"ny":ny,"cx":cx,"cy":cy,
                "len":length
            })

        self._dbg_counts = dict(Hough=n_hough, ROI_pass=n_roi, Front_pass=n_front, Len_pass=n_len,
                                Angle_pass=n_ang, Accepted=len(out), use_roi=use_roi)
        return out

    # ---------- Merge ----------
    def _merge_lines_bins(self, lines):
        if not lines: return []
        a_bin = max(1.0, float(self.merge_angle_tol_deg))  # deg
        r_bin = max(1e-3, float(self.merge_rho_tol_m))     # m
        buckets = {}
        for L in lines:
            ang_deg = (math.degrees(L["ang"]) + 360.0) % 180.0
            nx, ny = L["nx"], L["ny"]; cx, cy = L["cx"], L["cy"]
            s = cx*nx + cy*ny  # 법선 오프셋
            a_key = int(round(ang_deg / a_bin))
            r_key = int(round(s / r_bin))
            key = (a_key, r_key)
            if key not in buckets or L["len"] > buckets[key]["len"]:
                L2 = dict(L); L2["s"] = s
                buckets[key] = L2
        merged = list(buckets.values())
        merged.sort(key=lambda d: -d["len"])
        if len(merged) > self.max_lines_after_merge:
            merged = merged[:self.max_lines_after_merge]
        return merged

    # ---------- Pair scan: 평행성 + 폭 + 겹침 ----------
    @staticmethod
    def _delta_angle(a, b):
        # 병렬성: pi 주기
        d = abs(((a-b)+math.pi)%(2*math.pi)-math.pi)
        return d

    @staticmethod
    def _proj_overlap_on_angle(Li, Lj, ang):
        tx, ty = math.cos(ang), math.sin(ang)
        ip1 = Li["x1"]*tx + Li["y1"]*ty; ip2 = Li["x2"]*tx + Li["y2"]*ty
        jp1 = Lj["x1"]*tx + Lj["y1"]*ty; jp2 = Lj["x2"]*tx + Lj["y2"]*ty
        i_lo,i_hi = (ip1,ip2) if ip1<=ip2 else (ip2,ip1)
        j_lo,j_hi = (jp1,jp2) if jp1<=jp2 else (jp2,jp1)
        lo = max(i_lo, j_lo); hi = min(i_hi, j_hi)
        return max(0.0, hi-lo)

    def _pairs_parallel(self, lines):
        if not lines: return []
        pairs=[]; self._dbg_last_dists=[]
        tol_ang = math.radians(float(self.pair_parallel_tol_deg))
        n = len(lines)
        # O(n^2) but n<=max_lines_after_merge(<=80)
        for i in range(n):
            Li = lines[i]; ai = Li["ang"]; nxi, nyi = Li["nx"], Li["ny"]
            for j in range(i+1, n):
                Lj = lines[j]; aj = Lj["ang"]
                if self._delta_angle(ai, aj) > tol_ang: continue
                # 폭: Li 법선 기준 수직거리
                d = abs((Lj["cx"]-Li["cx"])*nyi - (Lj["cy"]-Li["cy"])*nxi)
                if len(self._dbg_last_dists) < 12: self._dbg_last_dists.append(round(d,3))
                ok_width=False
                if self.width_mode.lower()=="candidates":
                    ok_width = (np.min(np.abs(self.width_candidates - d)) <= self.pair_width_tol)
                else:
                    wmin, wmax = float(self.width_range_m[0]), float(self.width_range_m[1])
                    if wmin > wmax: wmin, wmax = wmax, wmin
                    ok_width = (wmin <= d <= wmax)
                if not ok_width: continue
                # 겹침길이: 평균각 기준 투영
                ang = 0.5*(ai+aj)
                ov = self._proj_overlap_on_angle(Li, Lj, ang)
                if ov < self.min_parallel_overlap_m: continue
                pairs.append((Li, Lj, d, ov))
                if len(pairs) >= self.max_pairs_considered: return pairs
        return pairs

    def _rect_from_pair(self, Li, Lj, width, length_cap):
        ang = 0.5*(Li["ang"] + Lj["ang"])
        cx  = 0.5*(Li["cx"] + Lj["cx"])
        cy  = 0.5*(Li["cy"] + Lj["cy"])
        ov  = self._proj_overlap_on_angle(Li, Lj, ang)
        length = min(length_cap, ov) if ov > 0 else length_cap
        return (cx, cy, ang, width, length, ov)

    def _occupancy_rect(self, rect):
        if not self.use_cloud_for_occupancy or self.last_cloud is None:
            return False, 0.0
        cx, cy, ang, w, l, _ = rect
        half_w = w/2.0; half_l = l/2.0
        xmin = cx - half_l; xmax = cx + half_l
        ymin = cy - half_w; ymax = cy + half_w
        occ_pts = 0; total=0
        for x,y,z in pc2.read_points(self.last_cloud, field_names=("x","y","z"), skip_nans=True):
            total += 1
            if (xmin<=x<=xmax) and (ymin<=y<=ymax) and (self.obstacle_z_low<=z<=self.obstacle_z_high):
                occ_pts += 1
        ratio = (occ_pts / max(1,total))
        return (ratio >= self.occupied_ratio_thr), ratio

    def _score_rect(self, w, length, overlap):
        if self.width_mode == "candidates":
            dw = float(np.min(np.abs(self.width_candidates - w)))
            sw = clamp(1.0 - dw / max(1e-6, self.pair_width_tol), 0.0, 1.0)
        else:
            wmin, wmax = float(self.width_range_m[0]), float(self.width_range_m[1])
            mid = 0.5*(wmin+wmax); half = max(1e-6, 0.5*(wmax-wmin))
            sw = clamp(1.0 - abs(w-mid)/half, 0.0, 1.0)
        sl = clamp(min(1.0, length / max(1e-6, self.slot_length)), 0.0, 1.0)
        so = clamp(min(1.0, overlap / max(1e-6, self.slot_length)), 0.0, 1.0)
        return max(self.min_score, float(0.55*sw + 0.30*sl + 0.15*so))

    # ---------- Publishing ----------
    def publish(self, rects):
        header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id)
        arr = ParkingSlotArray(); arr.header = header
        mk = MarkerArray()
        best_ps = None; best_score = -1.0; mid=0

        rects = rects[:self.max_slots_publish]
        for (cx, cy, ang, w, l, score) in rects:
            occupied, _ = self._occupancy_rect((cx,cy,ang,w,l,0.0))
            ps = ParkingSlot()
            ps.pose = Pose(position=Point(cx,cy,0.0), orientation=rot_to_quat(ang))
            ps.width=float(w); ps.length=float(l); ps.score=float(score); ps.occupied=bool(occupied)
            arr.slots.append(ps)

            if self.publish_markers:
                col = (0.85,0.15,0.15,0.9) if occupied else (0.10,0.85,0.10,0.9)
                mk.markers.append(make_marker_box(self.frame_id, mid, cx, cy, ang, w, l, 0.02, col)); mid+=1

            if (not occupied) and score>best_score:
                best_score=score; best_ps=ps

        self.pub_slots.publish(arr)
        if best_ps is not None:
            self.pub_best.publish(PoseStamped(header=header, pose=best_ps.pose))
        if self.publish_markers:
            self.pub_mk.publish(mk)

    # ---------- Image callback ----------
    def cb_image(self, msg):
        now = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()
        if self.max_processing_fps > 0 and now < self._next_ok_time: return
        if self.max_processing_fps > 0: self._next_ok_time = now + (1.0 / self.max_processing_fps)

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, "[vision] cv_bridge error: %s", str(e)); return

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if cv_img.ndim==3 else cv_img
        clahe = cv2.createCLAHE(clipLimit=self.gray_clip_limit, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        if self.adaptive:
            bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 21, -5)
        else:
            _, bin_img = cv2.threshold(gray, self.binary_thr, 255, cv2.THRESH_BINARY)
        bin_img = cv2.medianBlur(bin_img, 3)
        edges  = cv2.Canny(bin_img, self.canny_low_high[0], self.canny_low_high[1])

        # 1) 라인 검출 (+autorelax)
        lines_xy = self._lines_from_edges(edges, use_roi=self.use_roi)
        relaxed = False
        if len(lines_xy)==0 and self.autorelax_if_zero and self.use_roi:
            lines_xy = self._lines_from_edges(edges, use_roi=False); relaxed=True

        # 2) 병합
        merged = self._merge_lines_bins(lines_xy)

        # 3) 페어 구성(평행성 + 폭 + 겹침)
        pairs = self._pairs_parallel(merged)

        # 4) 슬롯 후보 + 스코어
        rects=[]
        for (Li, Lj, dw, ov) in pairs:
            (cx,cy,ang,w,l,ov2) = self._rect_from_pair(Li, Lj, width=dw, length_cap=self.slot_length)
            score = self._score_rect(w, l, ov2)
            rects.append((cx,cy,ang,w,l,score))
        rects.sort(key=lambda r: (-r[5], math.hypot(r[0], r[1])))

        self.publish(rects)

        dc = self._dbg_counts
        dists_str = ("  d-samples=" + str(self._dbg_last_dists)) if (len(pairs)==0 and len(self._dbg_last_dists)>0) else ""
        rospy.loginfo_throttle(
            1.0,
            "[vision] Hough=%d, ROI_pass=%d, Front_pass=%d, Len_pass=%d, Angle_pass=%d, Accepted=%d%s -> merged=%d, pairs=%d, slots(pub)=%d  ROI(x=[%.1f,%.1f],y=[%.1f,%.1f])%s",
            dc.get("Hough",0), dc.get("ROI_pass",0), dc.get("Front_pass",0), dc.get("Len_pass",0), dc.get("Angle_pass",0),
            dc.get("Accepted",0), " [RELAX]" if relaxed else "",
            len(merged), len(pairs), min(len(rects), self.max_slots_publish),
            self.roi_x_m[0], self.roi_x_m[1], self.roi_y_m[0], self.roi_y_m[1], dists_str
        )

def main():
    rospy.init_node("parking_vision_detector")
    VisionParking()
    rospy.spin()

if __name__ == "__main__":
    main()
