#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parking line detector (LiDAR→BEV→Edges→Hough→Parallel pairs→Slot scoring)

Changes:
 - NEW param ~publish_empty_heartbeat (default: False)
   When True, publishes empty /parking/slots at 5 Hz as heartbeat.
   Default is False to avoid wiping RViz markers.
 - Keep: show_pair_lines False by default is handled via launch.
"""

import rospy, numpy as np, math, functools, collections, threading, time, ast, traceback
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from auto_parking.msg import ParkingSlot, ParkingSlotArray
from tf.transformations import quaternion_from_euler

def integral_image(A):
    I = np.zeros((A.shape[0]+1, A.shape[1]+1), dtype=np.float32)
    I[1:,1:] = np.cumsum(np.cumsum(A.astype(np.float32), axis=0), axis=1)
    return I

def rect_sum(I, i0,j0, i1,j1):
    i0+=1; j0+=1; i1+=1; j1+=1
    return I[i1,j1]-I[i0-1,j1]-I[i1,j0-1]+I[i0-1,j0-1]

class LineDetector:
    def __init__(self):
        # frames / topics
        self.frame = rospy.get_param("~frame_id", "map")
        self.points_topic = rospy.get_param("~points_topic", "/ouster/points_map")
        self.use_wall = bool(rospy.get_param("~use_wall_thread", True))

        # heartbeat option (NEW)
        self.publish_empty_heartbeat = bool(rospy.get_param("~publish_empty_heartbeat", False))

        # debug line markers
        self.show_pair_lines = bool(rospy.get_param("~show_pair_lines", False))

        # BEV / accumulation
        self.R = float(rospy.get_param("~grid_radius_m", 7.0))
        self.res = float(rospy.get_param("~grid_res_m", 0.03))
        self.accum_s = float(rospy.get_param("~accumulate_sec", 1.0))
        self.points_hz_guess = float(rospy.get_param("~points_hz_guess", 6.5))
        self.max_scans = max(1, int(self.accum_s * self.points_hz_guess))

        # heights & bands
        self.obs_thr = float(rospy.get_param("~obstacle_min_height_m", 0.10))
        self.stop_low = float(rospy.get_param("~wheelstop_low_m", 0.05))
        self.stop_high= float(rospy.get_param("~wheelstop_high_m", 0.22))
        self.edge_z_band = float(rospy.get_param("~edge_z_band_m", 0.25))

        # edges
        self.edge_source = str(rospy.get_param("~edge_source", "density"))  # 'auto'|'intensity'|'density'
        self.intensity_keep_percentile = float(rospy.get_param("~intensity_keep_percentile", 85.0))
        self.edge_percentile = float(rospy.get_param("~edge_percentile", 97.0))
        self.density_edge_percentile = float(rospy.get_param("~density_edge_percentile", 88.0))

        # FLOOR mask
        self.floor_keep_percentile = float(rospy.get_param("~floor_keep_percentile", 25.0))
        self.min_floor_cover = float(rospy.get_param("~min_floor_cover", 0.30))

        # Hough
        default_deg = [-8,-5,-3,0,3,5,8, 82,85,87,90,93,95,98]
        raw = rospy.get_param("~hough_angles_deg", default_deg)
        if isinstance(raw, str):
            try: raw = ast.literal_eval(raw)
            except Exception: raw = default_deg
        self.thetas = np.deg2rad(np.asarray([float(d) for d in raw], dtype=np.float32))
        self.rho_res = float(rospy.get_param("~hough_rho_res_m", 0.10))
        self.min_votes_base = int(rospy.get_param("~hough_min_votes_base", 4))
        self.vote_frac = float(rospy.get_param("~hough_vote_frac", 0.003))
        self.top_k_per_theta = int(rospy.get_param("~hough_topk", 5))

        # slot geometry / sampling
        self.slot_l = float(rospy.get_param("~slot_length_m", 1.0))
        raw_w = rospy.get_param("~width_candidates_m", [0.48,0.50,0.52])
        if isinstance(raw_w, str):
            try: raw_w = ast.literal_eval(raw_w)
            except Exception: raw_w = [0.48,0.50,0.52]
        self.width_cands = [float(w) for w in raw_w]
        self.pair_width_tol = float(rospy.get_param("~pair_width_tol_m", 0.05))
        self.center_step = float(rospy.get_param("~center_step_m", 0.05))

        # scoring / occupancy
        self.min_score = float(rospy.get_param("~min_score", 0.22))
        self.w_empty = float(rospy.get_param("~w_empty", 0.70))
        self.w_wheelstop = float(rospy.get_param("~w_wheelstop", 0.20))
        self.w_edge = float(rospy.get_param("~w_edge", 0.15))
        self.occ_ratio_thr = float(rospy.get_param("~occupied_ratio_thr", 0.03))

        # front-only filter
        self.only_front  = bool(rospy.get_param("~only_front", True))
        self.front_axis  = str(rospy.get_param("~front_axis", "x"))  # 'x' or 'y'
        self.front_min_m = float(rospy.get_param("~front_min_m", 0.20))

        # buffers / pubs
        self.scan_buf = collections.deque(maxlen=self.max_scans)
        self.pub_slots = rospy.Publisher("/parking/slots", ParkingSlotArray, queue_size=10)
        self.pub_best  = rospy.Publisher("/parking/best_slot", PoseStamped, queue_size=10)
        self.pub_mark  = rospy.Publisher("/parking/debug/markers", MarkerArray, queue_size=10)

        # subscribe
        rospy.Subscriber(self.points_topic, PointCloud2,
                         functools.partial(self.cb_points, source=self.points_topic), queue_size=1)
        rospy.loginfo("[line] subscribing %s, frame=%s", self.points_topic, self.frame)

        # threads / timers
        if self.publish_empty_heartbeat:
            threading.Thread(target=self._hb_loop, daemon=True).start()
        if self.use_wall:
            threading.Thread(target=self._wall_loop, daemon=True).start()
        else:
            self._timer = rospy.Timer(rospy.Duration(0.20), self._tick_guard)

        # clear old markers if pair-lines hidden
        if not self.show_pair_lines:
            self.clear_markers(Header(stamp=rospy.Time.now(), frame_id=self.frame))

    def _hb_loop(self):
        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            self.pub_slots.publish(ParkingSlotArray(header=Header(stamp=rospy.Time.now(), frame_id=self.frame), slots=[]))
            r.sleep()

    def _wall_loop(self):
        while not rospy.is_shutdown():
            self._tick_guard(None); time.sleep(0.20)

    def _tick_guard(self, _):
        try: self.tick()
        except Exception as e:
            rospy.logerr_throttle(1.0, "[line] tick error: %s\n%s", repr(e), traceback.format_exc())

    def cb_points(self, msg: PointCloud2, source: str):
        fields = [f.name for f in msg.fields]
        has_i = ("intensity" in fields) or ("reflectivity" in fields)
        i_name = "intensity" if "intensity" in fields else ("reflectivity" if "reflectivity" in fields else None)
        if has_i:
            P = np.array(list(pc2.read_points(msg, field_names=("x","y","z",i_name), skip_nans=True)), dtype=np.float32)
        else:
            P = np.array(list(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)), dtype=np.float32)
            if P.size == 0: return
            P = np.hstack([P, np.zeros((P.shape[0],1), dtype=np.float32)])
        if P.size == 0: return
        self.scan_buf.append(P)

    # --- 이하 tick(), build_edge(), hough(), make_pairs(), generate_and_score(),
    #     score_box(), make_pair_markers(), clear_markers() 는 이전 버전과 동일 ---
    # (직전 메시지에서 제공된 구현 그대로 사용)
    # ---- 복붙 시작 ----
    def tick(self):
        if not self.scan_buf:
            rospy.logwarn_throttle(2.0, "[line] no points yet"); return

        P = np.vstack(list(self.scan_buf))
        x,y,z,I = P[:,0],P[:,1],P[:,2],P[:,3]

        R,res = self.R,self.res
        m = (x>-R)&(x<R)&(y>-R)&(y<R)
        x,y,z,I = x[m],y[m],z[m],I[m]
        N = x.size
        if N < 800:
            rospy.logwarn_throttle(1.0, "[line] few points in ROI: N=%d", N); return

        zhist, zedges = np.histogram(z, bins=60, range=(np.percentile(z,1), np.percentile(z,99)))
        z0 = 0.5*(zedges[np.argmax(zhist)] + zedges[np.argmax(zhist)+1])

        obs_mask = (z > z0 + self.obs_thr)
        ws_mask  = (z > z0 + self.stop_low) & (z < z0 + self.stop_high)
        band_mask = (np.abs(z - z0) <= self.edge_z_band)

        G = int((2*R)/res) + 1
        ix = ((x + R)/res).astype(np.int32)
        iy = ((y + R)/res).astype(np.int32)
        valid = (ix>=0)&(ix<G)&(iy>=0)&(iy<G)

        occ = np.zeros((G,G), dtype=np.uint16)
        wsg = np.zeros((G,G), dtype=np.uint16)
        np.add.at(occ, (ix[valid & obs_mask], iy[valid & obs_mask]), 1)
        np.add.at(wsg, (ix[valid & ws_mask],  iy[valid & ws_mask]), 1)

        v_band = valid & band_mask
        ix_b, iy_b = ix[v_band], iy[v_band]
        I_b = I[v_band]

        sumI = np.zeros((G,G), dtype=np.float32)
        cntI = np.zeros((G,G), dtype=np.float32)
        if I_b.size:
            np.add.at(sumI, (ix_b,iy_b), I_b); np.add.at(cntI, (ix_b,iy_b), 1.0)
        imgI = np.divide(sumI, np.maximum(1.0, cntI))
        if np.max(imgI) > 0: imgI = imgI / (np.max(imgI)+1e-6)

        imgD = np.zeros((G,G), dtype=np.float32)
        if ix_b.size: np.add.at(imgD, (ix_b,iy_b), 1.0)
        if np.max(imgD) > 0: imgD = imgD / (np.max(imgD)+1e-6)

        nz = imgD[imgD>0]
        floor_thr = np.percentile(nz, self.floor_keep_percentile) if nz.size else 1.0
        floor_mask = (imgD >= floor_thr).astype(np.uint8)
        floor_int = integral_image(floor_mask)

        E, src, edge_px, edge_mag = self.build_edge(imgI, imgD)

        lines, min_votes = self.hough(E, R, res)

        pairs = self.make_pairs(lines, self.width_cands, self.pair_width_tol)

        slots = self.generate_and_score(occ, wsg, edge_mag, floor_int, pairs, R, res)

        hdr = Header(stamp=rospy.Time.now(), frame_id=self.frame)
        arr = ParkingSlotArray(); arr.header = hdr; arr.slots=[]
        best=None; best_score=-1.0

        for (cx,cy,yaw,W,score,occ_ratio) in slots:
            s = ParkingSlot()
            s.pose.position.x=cx; s.pose.position.y=cy; s.pose.position.z=0.0
            s.pose.orientation = Quaternion(*quaternion_from_euler(0,0,yaw))
            s.width=float(W); s.length=float(self.slot_l)
            s.score=float(score); s.occupied = bool(occ_ratio > self.occ_ratio_thr)
            arr.slots.append(s)
            if score>best_score: best_score=score; best=(cx,cy,yaw)

        self.pub_slots.publish(arr)
        if best is not None:
            msg = PoseStamped(); msg.header = hdr
            msg.pose.position.x, msg.pose.position.y = best[0], best[1]
            msg.pose.orientation = Quaternion(*quaternion_from_euler(0,0,best[2]))
            self.pub_best.publish(msg)

        rospy.loginfo_throttle(
            1.0,
            "[line] ROI_N=%d edge_src=%s edge_px=%d min_votes=%d lines=%d pairs=%d candidates=%d best=%.3f floor_thr=%.3f",
            int(N), src, int(edge_px), int(min_votes), len(lines), len(pairs), len(slots),
            (best_score if best is not None else -1.0), float(floor_thr)
        )

        if self.show_pair_lines:
            self.pub_mark.publish(self.make_pair_markers(pairs, R, res, hdr))
        else:
            self.clear_markers(hdr)

    def build_edge(self, imgI, imgD):
        def grad(A):
            gx = np.abs(np.diff(A, axis=0, prepend=A[:1,:]))
            gy = np.abs(np.diff(A, axis=1, prepend=A[:,:1]))
            G = gx + gy
            if np.max(G)>0: G = G/(np.max(G)+1e-6)
            return G

        frac_int = float(np.count_nonzero(imgI>0.02))/float(imgI.size+1e-6)
        use_int = (self.edge_source=="intensity") or (self.edge_source=="auto" and frac_int>0.05)

        if use_int:
            vals = imgI[imgI>0]
            keep_thr = np.percentile(vals, self.intensity_keep_percentile) if vals.size else 1.0
            mask = (imgI>=keep_thr).astype(np.float32)
            G = grad(imgI)*mask
            thr = np.percentile(G[G>0], self.edge_percentile) if np.any(G>0) else 1.0
            E = (G>=thr).astype(np.uint8); src="intensity"
        else:
            G = grad(imgD)
            thr = np.percentile(G[G>0], self.density_edge_percentile) if np.any(G>0) else 1.0
            E = (G>=thr).astype(np.uint8); src="density"
        return E, src, int(E.sum()), G

    def hough(self, E, R, res):
        ys, xs = np.nonzero(E)
        if xs.size==0: return [], self.min_votes_base
        xs_m = -R + (xs+0.5)*res; ys_m = -R + (ys+0.5)*res
        rho_max = math.sqrt(2.0)*R + 0.5
        rho_bins = int((2*rho_max)/self.rho_res)+1; rho_base = -rho_max
        edge_px = xs_m.size
        min_votes = max(self.min_votes_base, int(self.vote_frac*edge_px))
        lines=[]
        for th in self.thetas:
            c,s = math.cos(th), math.sin(th)
            r = xs_m*c + ys_m*s
            q = np.floor((r - rho_base)/self.rho_res).astype(np.int32)
            valid = (q>=0)&(q<rho_bins)
            if not np.any(valid): continue
            hist = np.bincount(q[valid], minlength=rho_bins)
            if hist.max() < min_votes: continue
            idxs = np.argpartition(-hist, min(self.top_k_per_theta, rho_bins-1))[:self.top_k_per_theta]
            idxs = idxs[np.argsort(-hist[idxs])]
            for qi in idxs:
                if hist[qi] >= min_votes:
                    rho = rho_base + qi*self.rho_res
                    lines.append((th, rho, int(hist[qi])))
        return lines, min_votes

    def make_pairs(self, lines, width_list, tol):
        pairs=[]
        for i in range(len(lines)):
            th_i,rho_i,v_i = lines[i]
            for j in range(i+1,len(lines)):
                th_j,rho_j,v_j = lines[j]
                dth = abs((th_i - th_j) * 180.0/math.pi); dth = min(dth, 180.0-dth)
                if dth > 8.0: continue
                sep = abs(rho_i - rho_j)
                width_used=None
                for w in width_list:
                    if abs(sep - w) <= max(tol, 0.15*w):
                        width_used=w; break
                if width_used is None: continue
                th = 0.5*(th_i + th_j); rho_mid = 0.5*(rho_i + rho_j)
                pairs.append((th, rho_mid, width_used, v_i+v_j))
        pairs.sort(key=lambda t: -t[3])
        return pairs[:12]

    def generate_and_score(self, occ, wsg, edge_mag, floor_int, pairs, R, res):
        G = occ.shape[0]
        step = max(1, int(self.center_step/res))
        ii = np.arange(0,G,step); jj = np.arange(0,G,step)
        occ_int = integral_image((occ>0).astype(np.uint8))
        wsg_int = integral_image((wsg>0).astype(np.uint8))
        edge_int= integral_image(edge_mag.astype(np.float32))

        def cell_xy(i,j): return (-R + (i+0.5)*res, -R + (j+0.5)*res)

        out=[]
        for (th, rho_mid, W, votes) in pairs:
            c,s = math.cos(th), math.sin(th)
            for i in ii:
                for j in jj:
                    x,y = cell_xy(i,j)
                    if self.only_front:
                        if self.front_axis == "x" and x < self.front_min_m: 
                            continue
                        if self.front_axis == "y" and y < self.front_min_m: 
                            continue
                    d = x*c + y*s - rho_mid
                    if abs(d) > (W*0.8): continue
                    score, occ_ratio, floor_cov = self.score_box(
                        occ_int, wsg_int, edge_int, floor_int,
                        x,y, th, W,self.slot_l, band_th=0.35, R=R, res=res
                    )
                    if floor_cov < self.min_floor_cover: 
                        continue
                    if score >= self.min_score:
                        out.append((x,y,th,W,score,occ_ratio))

        out.sort(key=lambda t: -t[4])
        kept=[]
        def iouAABB(a,b):
            ax,ay,ath,aw = a[0],a[1],a[2],a[3]; bx,by,bth,bw = b[0],b[1],b[2],b[3]
            def aabb(cx,cy,yaw,w,l):
                cs,sn=math.cos(yaw),math.sin(yaw); xs=[]; ys=[]
                for dx,dy in [(-l/2,-w/2),(-l/2,w/2),(l/2,w/2),(l/2,-w/2)]:
                    x = cx + cs*dx - sn*dy; y = cy + sn*dx + cs*dy; xs.append(x); ys.append(y)
                return min(xs),min(ys),max(xs),max(ys)
            A=aabb(ax,ay,ath,aw,self.slot_l); B=aabb(bx,by,bth,bw,self.slot_l)
            xA=max(A[0],B[0]); yA=max(A[1],B[1]); xB=min(A[2],B[2]); yB=min(A[3],B[3])
            inter=max(0.0,xB-xA)*max(0.0,yB-yA)
            areaA=(A[2]-A[0])*(A[3]-A[1]); areaB=(B[2]-B[0])*(B[3]-B[1])
            return inter/(areaA+areaB-inter+1e-6)
        for cand in out:
            if all(iouAABB(cand,k)<=0.3 for k in kept): kept.append(cand)
            if len(kept)>=12: break
        return kept

    def score_box(self, occ_int, wsg_int, edge_int, floor_int, cx,cy,yaw,W,L, band_th, R, res):
        cs,sn = math.cos(yaw), math.sin(yaw)
        hw=W/2.0; hl=L/2.0
        xs=[]; ys=[]
        for dx,dy in [(-hl,-hw),(-hl,hw),(hl,hw),(hl,-hw)]:
            x=cx + cs*dx - sn*dy; y=cy + sn*dx + cs*dy; xs.append(x); ys.append(y)
        xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)

        def xy2idx(x,y): return (int((x+R)/res), int((y+R)/res))
        imin,jmin = xy2idx(xmin,ymin); imax,jmax = xy2idx(xmax,ymax)
        G = occ_int.shape[0]-1
        if imax<0 or jmax<0 or imin>G-1 or jmin>G-1: return (0.0,1.0,0.0)
        imin=max(0,min(G-1,imin)); imax=max(0,min(G-1,imax))
        jmin=max(0,min(G-1,jmin)); jmax=max(0,min(G-1,jmax))

        step=max(1, int(0.18/res))
        total=0; occ_hits=0; edge_sum=0.0; floor_hits=0
        for i in range(imin, imax+1, step):
            for j in range(jmin, jmax+1, step):
                x=-R+(i+0.5)*res; y=-R+(j+0.5)*res
                dx=x-cx; dy=y-cy
                u= cs*dx + sn*dy; v=-sn*dx + cs*dy
                if abs(u)<=hl and abs(v)<=hw:
                    total+=1
                    if rect_sum(occ_int,i,j,i,j)>0: occ_hits+=1
                    if rect_sum(floor_int,i,j,i,j)>0: floor_hits+=1
                    edge_sum+=rect_sum(edge_int,i,j,i,j)
        if total==0: return (0.0,1.0,0.0)
        empty_ratio = 1.0 - (occ_hits/float(total))
        floor_cov = floor_hits/float(total)

        bc=0; bh=0
        for i in range(imin, imax+1, step):
            for j in range(jmin, jmax+1, step):
                x=-R+(i+0.5)*res; y=-R+(j+0.5)*res
                dx=x-cx; dy=y-cy
                u= cs*dx + sn*dy; v=-sn*dx + cs*dy
                if (hl-band_th)<=u<=hl and abs(v)<=hw:
                    bc+=1
                    if rect_sum(wsg_int,i,j,i,j)>0: bh+=1
        wheelstop = (bh/float(bc)) if bc>0 else 0.0

        edge_support = np.clip(edge_sum/float(total), 0.0, 1.0)
        score = (self.w_empty*empty_ratio + self.w_wheelstop*wheelstop + self.w_edge*edge_support)
        occ_ratio = (occ_hits/float(total))
        return (float(score), float(occ_ratio), float(floor_cov))

    def make_pair_markers(self, pairs, R, res, hdr):
        arr = MarkerArray(); mid=0
        for (th,rho_mid,W,votes) in pairs:
            c,s = math.cos(th), math.sin(th)
            tdir = np.array([-s, c]); p0 = np.array([c*rho_mid, s*rho_mid])
            ts = np.linspace(-R,R,10)
            pts = p0[None,:] + ts[:,None]*tdir[None,:]
            m = (pts[:,0]>=-R)&(pts[:,0]<=R)&(pts[:,1]>=-R)&(pts[:,1]<=R)
            pts = pts[m]
            if pts.shape[0]<2: continue
            from geometry_msgs.msg import Point
            mk=Marker(); mk.header=hdr; mk.ns=f"pairs_w{W:.2f}"; mk.id=mid; mid+=1
            mk.type=Marker.LINE_STRIP; mk.action=Marker.ADD; mk.scale.x=0.05
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = 0.2, 0.8, 1.0, 0.9
            mk.lifetime = rospy.Duration(0.5)
            for k in range(pts.shape[0]):
                p=Point(); p.x=float(pts[k,0]); p.y=float(pts[k,1]); p.z=0.03; mk.points.append(p)
            arr.markers.append(mk)
        return arr

    def clear_markers(self, hdr):
        ma = MarkerArray()
        m = Marker(); m.header = hdr; m.action = Marker.DELETEALL
        ma.markers.append(m)
        self.pub_mark.publish(ma)

if __name__ == "__main__":
    rospy.init_node("parking_line_detector")
    LineDetector()
    rospy.spin()
