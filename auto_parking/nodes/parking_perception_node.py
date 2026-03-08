#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR-only parking slot detection with wall-clock heartbeat/processing.
- Always publishes /parking/slots ~5Hz (heartbeat), but '빈배열 덮어쓰기' 방지 로직 추가
- Points topic: auto-select (/ouster/points_map preferred) or fixed via ~points_topic
- 풍부한 디버그 로그: 콜백 진입/BEV 포인트 수/후보 개수/최고 점수
"""

import rospy, numpy as np, math, collections, ast, functools, threading, time, traceback
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Header
from auto_parking.msg import ParkingSlot, ParkingSlotArray
from tf.transformations import quaternion_from_euler

# ---------- utilities ----------
def integral_image(A):
    I = np.zeros((A.shape[0]+1, A.shape[1]+1), dtype=np.float32)
    I[1:,1:] = np.cumsum(np.cumsum(A.astype(np.float32), axis=0), axis=1)
    return I

def rect_sum(I, i0,j0, i1,j1):
    i0+=1; j0+=1; i1+=1; j1+=1
    return I[i1,j1]-I[i0-1,j1]-I[i1,j0-1]+I[i0-1,j0-1]

def rect_iou(r1, r2):
    c1x,c1y,t1,w1,l1 = r1; c2x,c2y,t2,w2,l2 = r2
    def aabb(cx,cy,yaw,w,l):
        cs,sn=math.cos(yaw),math.sin(yaw)
        xs=[]; ys=[]
        for dx,dy in [(-l/2,-w/2),(-l/2, w/2),(l/2, w/2),(l/2,-w/2)]:
            x = cx + cs*dx - sn*dy; y = cy + sn*dx + cs*dy
            xs.append(x); ys.append(y)
        return min(xs),min(ys),max(xs),max(ys)
    a1=aabb(c1x,c1y,t1,w1,l1); a2=aabb(c2x,c2y,t2,w2,l2)
    xA=max(a1[0],a2[0]); yA=max(a1[1],a2[1]); xB=min(a1[2],a2[2]); yB=min(a1[3],a2[3])
    inter=max(0.0,xB-xA)*max(0.0,yB-yA)
    A1=(a1[2]-a1[0])*(a1[3]-a1[1]); A2=(a2[2]-a2[0])*(a2[3]-a2[1])
    return inter/(A1+A2-inter+1e-6)

# ---------- detector ----------
class BEVDetector:
    def __init__(self):
        # Params
        self.frame        = rospy.get_param("~frame_id", "map")
        self.points_param = rospy.get_param("~points_topic", "auto")  # "auto" or topic name
        self.use_wall     = rospy.get_param("~use_wall_thread", True)

        self.slot_w   = rospy.get_param("~slot_width_m", 2.5)
        self.slot_l   = rospy.get_param("~slot_length_m", 5.0)
        self.w_tol    = rospy.get_param("~slot_width_tol", 0.4)
        self.l_tol    = rospy.get_param("~slot_length_tol", 0.7)
        self.grid_res = rospy.get_param("~grid_res_m", 0.10)
        self.grid_r   = rospy.get_param("~grid_radius_m", 12.0)
        self.accum_s  = rospy.get_param("~accumulate_sec", 0.5)
        self.max_scans= max(1, int(self.accum_s * rospy.get_param("~points_hz_guess", 10.0)))
        self.obs_thr  = rospy.get_param("~obstacle_min_height_m", 0.12)
        self.stop_low = rospy.get_param("~wheelstop_low_m", 0.08)
        self.stop_high= rospy.get_param("~wheelstop_high_m", 0.25)

        default_deg   = [-10, -5, 0, 5, 10, 85, 90, 95]
        raw = rospy.get_param("~yaw_candidates_deg", default_deg)
        if isinstance(raw, (list, tuple, np.ndarray)):
            deg_list = [float(x) for x in raw]
        elif isinstance(raw, str):
            try:
                pr = ast.literal_eval(raw)
                deg_list = [float(x) for x in pr] if isinstance(pr, (list, tuple, np.ndarray)) \
                    else [float(x) for x in raw.replace('[','').replace(']','').split(',') if x.strip()]
            except Exception:
                deg_list = default_deg
        else:
            deg_list = default_deg
        self.yaw_guesses = np.deg2rad(np.asarray(deg_list, dtype=np.float32))

        self.center_step = rospy.get_param("~center_step_m", 0.25)
        self.w_empty     = rospy.get_param("~w_empty", 0.65)
        self.w_wheelstop = rospy.get_param("~w_wheelstop", 0.25)
        self.w_intensity = rospy.get_param("~w_intensity", 0.10)
        self.min_score   = rospy.get_param("~min_score", 0.45)
        self.occ_ratio_thr = rospy.get_param("~occupied_ratio_thr", 0.02)

        # Buffers & state
        self.scan_buf = collections.deque(maxlen=self.max_scans)
        self.active_source = None

        # Publishers
        self.pub_slots = rospy.Publisher("/parking/slots", ParkingSlotArray, queue_size=10, latch=False)
        self.pub_best  = rospy.Publisher("/parking/best_slot", PoseStamped, queue_size=10)

        # Subscriptions
        if self.points_param == "auto":
            rospy.Subscriber("/ouster/points_map", PointCloud2,
                             functools.partial(self.cb_points, source="/ouster/points_map"), queue_size=1)
            rospy.Subscriber("/ouster/points", PointCloud2,
                             functools.partial(self.cb_points, source="/ouster/points"), queue_size=1)
            rospy.loginfo("[perception] auto topic select: waiting /ouster/points_map or /ouster/points ...")
        else:
            self.active_source = self.points_param
            rospy.Subscriber(self.points_param, PointCloud2,
                             functools.partial(self.cb_points, source=self.points_param), queue_size=1)
            rospy.loginfo("[perception] fixed points_topic=%s", self.points_param)

        # Heartbeat gate (빈배열이 실데이터를 덮지 않도록)
        self._last_slots_pub_time = 0.0
        self._last_slots_count = 0
        self._hb_lock = threading.Lock()

        # Processing loop
        if self.use_wall:
            rospy.loginfo("[perception] using WALL-CLOCK thread (ignores /use_sim_time)")
            self._thr = threading.Thread(target=self._wall_loop, daemon=True)
            self._thr.start()
        else:
            rospy.loginfo("[perception] using ROS Timer")
            self._timer = rospy.Timer(rospy.Duration(0.20), self._tick_guard)

        # Independent heartbeat thread (slots 빈 배열 5Hz 보장, 단 최근에 유효 슬롯이 있으면 억제)
        self._hb_thr = threading.Thread(target=self._hb_loop, daemon=True)
        self._hb_thr.start()

        rospy.loginfo("[perception] frame=%s, yaw_deg=%s", self.frame, ",".join([str(int(d)) for d in deg_list]))

    # ---- heartbeat thread (with gating) ----
    def _hb_loop(self):
        rate = rospy.Rate(5)  # 5Hz
        while not rospy.is_shutdown():
            with self._hb_lock:
                now = rospy.Time.now().to_sec()
                # 최근 0.6초 내 유효 슬롯을 발행했다면 빈배열 하트비트 억제
                if self._last_slots_count > 0 and (now - self._last_slots_pub_time) < 0.6:
                    rate.sleep()
                    continue
            hdr = Header(stamp=rospy.Time.now(), frame_id=self.frame)
            self.pub_slots.publish(ParkingSlotArray(header=hdr, slots=[]))
            rospy.loginfo_throttle(1.0, "[perception] published empty heartbeat (t=%.3f)", hdr.stamp.to_sec())
            rate.sleep()

    # ---- wall clock loop ----
    def _wall_loop(self):
        while not rospy.is_shutdown():
            self._tick_guard(None)
            time.sleep(0.20)  # 5Hz wall clock

    def _tick_guard(self, evt):
        try:
            self.tick(evt)
        except Exception as e:
            rospy.logerr_throttle(1.0, "[perception] tick error: %s\n%s", repr(e), traceback.format_exc())

    # ---- points callback ----
    def cb_points(self, msg: PointCloud2, source: str):
        rospy.loginfo_throttle(1.0, "[perception] cb_points called from %s (stamp=%.3f)",
                               source, msg.header.stamp.to_sec())

        if self.active_source is None:
            self.active_source = source
            rospy.loginfo("[perception] selected points topic: %s (frame=%s)", source, msg.header.frame_id)
        if source != self.active_source:
            return

        fields = [f.name for f in msg.fields]
        has_i = ("intensity" in fields) or ("reflectivity" in fields)
        i_name = "intensity" if "intensity" in fields else ("reflectivity" if "reflectivity" in fields else None)

        if has_i:
            pts = np.array(list(pc2.read_points(msg, field_names=("x","y","z",i_name), skip_nans=True)), dtype=np.float32)
        else:
            pts = np.array(list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)), dtype=np.float32)
            pts = np.hstack([pts, np.zeros((pts.shape[0],1),dtype=np.float32)])

        if pts.size == 0:
            return
        self.scan_buf.append((pts, has_i))

    # ---- main tick ----
    def tick(self, _evt):
        now = rospy.Time.now()
        hdr = Header(stamp=now, frame_id=self.frame)

        if len(self.scan_buf) == 0:
            rospy.logwarn_throttle(2.0, "[perception] no pointcloud yet (active source=%s)", self.active_source)
            return

        # 누적 포인트 병합
        pts_list = [b[0] for b in self.scan_buf]
        hasI = any(b[1] for b in self.scan_buf)
        P = np.vstack(pts_list)  # [N,4]
        x,y,z,I = P[:,0], P[:,1], P[:,2], P[:,3]

        # BEV crop (센터가 (0,0) 기준임: map 또는 os_sensor 좌표)
        R = self.grid_r
        m = (x>-R)&(x<R)&(y>-R)&(y<R)
        x,y,z,I = x[m], y[m], z[m], I[m]
        N = x.size
        rospy.loginfo_throttle(1.0, "[perception] BEV points after crop: N=%d (R=%.1f,res=%.2f,acc=%.2fs,buf=%d)",
                               int(N), R, self.grid_res, self.accum_s, len(self.scan_buf))
        if N < 80:
            rospy.logwarn_throttle(1.0, "[perception] too few points in BEV (N=%d) — try smaller grid_radius_m or longer accumulate_sec", int(N))
            return

        # Ground estimate
        zhist, zedges = np.histogram(z, bins=60, range=(np.percentile(z,1), np.percentile(z,99)))
        zbin = np.argmax(zhist)
        z0 = 0.5*(zedges[zbin]+zedges[zbin+1])

        # Classes
        obs_mask = (z > z0 + self.obs_thr)
        ws_mask  = (z > z0 + self.stop_low) & (z < z0 + self.stop_high)

        # Grid build
        res = self.grid_res
        G = int((2*R)/res) + 1
        ix = ((x + R)/res).astype(np.int32)
        iy = ((y + R)/res).astype(np.int32)
        valid = (ix>=0)&(ix<G)&(iy>=0)&(iy<G)
        ix, iy = ix[valid], iy[valid]
        I = I[valid]; obs_mask = obs_mask[valid]; ws_mask = ws_mask[valid]

        occ = np.zeros((G,G), dtype=np.uint8); np.add.at(occ, (ix[obs_mask], iy[obs_mask]), 1)
        wsg = np.zeros((G,G), dtype=np.uint8); np.add.at(wsg, (ix[ws_mask],  iy[ws_mask]),  1)

        # Intensity optional edge
        inten = None
        if hasI:
            sumI = np.zeros((G,G), dtype=np.float32)
            cntI = np.zeros((G,G), dtype=np.float32)
            np.add.at(sumI, (ix,iy), I); np.add.at(cntI, (ix,iy), 1.0)
            inten = np.divide(sumI, np.maximum(1.0, cntI))
            if np.max(inten) > 0: inten = inten/(np.max(inten)+1e-6)

        # 후보 스캔
        slots = self.scan_rectangles(occ, wsg, inten, R, res)
        rospy.loginfo_throttle(1.0, "[perception] candidates=%d %s",
                               len(slots), ("" if len(slots)==0 else "best=%.3f"% (max(s[5] for s in slots))))

        # Publish results
        arr = ParkingSlotArray()
        arr.header = hdr
        arr.slots = []

        best = None; best_score = -1.0
        for (cx, cy, yaw, w, l, score, occ_ratio) in slots:
            ps = ParkingSlot()
            ps.pose.position.x = cx; ps.pose.position.y = cy; ps.pose.position.z = 0.0
            q = quaternion_from_euler(0,0,yaw)
            ps.pose.orientation = Quaternion(*q)
            ps.width = w; ps.length = l; ps.score = float(score)
            ps.occupied = bool(occ_ratio > self.occ_ratio_thr)
            arr.slots.append(ps)
            if score > best_score:
                best_score = score; best = (cx, cy, yaw)

        self.pub_slots.publish(arr)
        with self._hb_lock:
            self._last_slots_pub_time = hdr.stamp.to_sec()
            self._last_slots_count = len(arr.slots)

        if best is not None:
            best_msg = PoseStamped()
            best_msg.header = hdr
            best_msg.pose.position.x, best_msg.pose.position.y = best[0], best[1]
            best_msg.pose.orientation = Quaternion(*quaternion_from_euler(0,0,best[2]))
            self.pub_best.publish(best_msg)

    # ---- rectangle scan & scoring ----
    def scan_rectangles(self, occ, wsg, inten, R, res):
        G = occ.shape[0]
        def cell_to_xy(i,j): return (-R + i*res, -R + j*res)
        step = max(1, int(self.center_step / res))
        ii = np.arange(0, G, step); jj = np.arange(0, G, step)
        centers = [(i,j) for i in ii for j in jj]

        W=self.slot_w; L=self.slot_l
        kept=[]
        occ_int = integral_image((occ>0).astype(np.uint8))
        wsg_int = integral_image((wsg>0).astype(np.uint8))

        if inten is not None:
            gx = np.abs(np.diff(inten, axis=0, prepend=inten[:1,:]))
            gy = np.abs(np.diff(inten, axis=1, prepend=inten[:,:1]))
            edge = np.clip(gx+gy, 0, 1)
            edge_int = integral_image(edge.astype(np.float32))
        else:
            edge_int = None

        for yaw in self.yaw_guesses:
            topK=[]
            for (ci,cj) in centers:
                cx,cy = cell_to_xy(ci,cj)
                score, occ_ratio = self.score_box(occ_int, wsg_int, edge_int, cx, cy, yaw, W, L, 0.35, R, res)
                if score < self.min_score: continue
                topK.append((score, occ_ratio, cx, cy, yaw))
            topK.sort(key=lambda t: -t[0])
            kept.extend(topK[:30])

        kept.sort(key=lambda t: -t[0])
        final=[]
        for cand in kept:
            score, occ_ratio, cx, cy, yaw = cand
            ok=True
            for f in final:
                _,_,fx,fy,fyaw = f
                if rect_iou((cx,cy,yaw,W,L),(fx,fy,fyaw,W,L)) > 0.3:
                    ok=False; break
            if ok: final.append(cand)
            if len(final)>=10: break

        return [(c[2],c[3],c[4],W,L,c[0],c[1]) for c in final]

    def score_box(self, occ_int, wsg_int, edge_int, cx,cy,yaw,W,L,band_th,R,res):
        cs,sn = math.cos(yaw), math.sin(yaw)
        half_w=W/2.0; half_l=L/2.0
        corners=[]
        for dx,dy in [(-half_l,-half_w),(-half_l,half_w),(half_l,half_w),(half_l,-half_w)]:
            x=cx + cs*dx - sn*dy; y=cy + sn*dx + cs*dy; corners.append((x,y))
        xs=[p[0] for p in corners]; ys=[p[1] for p in corners]
        xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)

        def xy_to_idx(x,y): return (int((x+R)/res), int((y+R)/res))
        imin,jmin = xy_to_idx(xmin,ymin); imax,jmax = xy_to_idx(xmax,ymax)
        G = occ_int.shape[0]-1
        if imax<0 or jmax<0 or imin>G-1 or jmin>G-1: return (0.0,1.0)
        imin=max(0,min(G-1,imin)); imax=max(0,min(G-1,imax))
        jmin=max(0,min(G-1,jmin)); jmax=max(0,min(G-1,jmax))

        step=max(1, int(0.20/res))
        total=0; occ_hits=0; edge_sum=0.0
        for i in range(imin, imax+1, step):
            for j in range(jmin, jmax+1, step):
                x=-R+(i+0.5)*res; y=-R+(j+0.5)*res
                dx=x-cx; dy=y-cy
                u= cs*dx + sn*dy; v=-sn*dx + cs*dy
                if abs(u)<=half_l and abs(v)<=half_w:
                    total+=1
                    if rect_sum(occ_int,i,j,i,j)>0: occ_hits+=1
                    if edge_int is not None: edge_sum+=rect_sum(edge_int,i,j,i,j)

        if total==0: return (0.0,1.0)
        empty_ratio = 1.0 - (occ_hits/float(total))

        band_cells=0; band_hits=0
        for i in range(imin, imax+1, step):
            for j in range(jmin, jmax+1, step):
                x=-R+(i+0.5)*res; y=-R+(j+0.5)*res
                dx=x-cx; dy=y-cy
                u= cs*dx + sn*dy; v=-sn*dx + cs*dy
                if (half_l-band_th)<=u<=half_l and abs(v)<=half_w:
                    band_cells+=1
                    if rect_sum(wsg_int,i,j,i,j)>0: band_hits+=1
        wheelstop = (band_hits/float(band_cells)) if band_cells>0 else 0.0

        edge_support=0.0
        if edge_int is not None and total>0:
            edge_support = np.clip(edge_sum/float(total), 0.0, 1.0)

        score = (self.w_empty*empty_ratio
                 + self.w_wheelstop*wheelstop
                 + self.w_intensity*edge_support)
        occ_ratio = (occ_hits/float(total))
        return (float(score), float(occ_ratio))

# -------- main --------
if __name__ == "__main__":
    rospy.init_node("parking_perception")
    BEVDetector()
    rospy.spin()
