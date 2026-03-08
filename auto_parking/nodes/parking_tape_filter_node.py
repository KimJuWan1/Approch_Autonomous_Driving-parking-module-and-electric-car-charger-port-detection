#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiDAR-only tape extractor by intensity/reflectivity.
- Input : PointCloud2 (e.g., /ouster/points_map)
- Output: ~points (filtered tape points)            [PointCloud2]
         ~markers (clusters & pca line debug)      [MarkerArray]
옵션으로 'boost=true'면 클러스터 주성분 축을 따라 얇은 포인트를 추가 생성해
선 검출(Hough/density)에 표가 잘 몰리도록 돕습니다(카메라 필요 없음).
"""

import rospy, math, numpy as np
from collections import defaultdict, deque
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

def _has_field(msg, name): return any(f.name == name for f in msg.fields)

def _cloud_xyz32(points, frame, stamp):
    hdr = Header(frame_id=frame, stamp=stamp)
    return pc2.create_cloud_xyz32(hdr, points)

class TapeFilterNode:
    def __init__(self):
        # ---- 기본 입/출력 ----
        self.frame_id    = rospy.get_param("~frame_id", "map")
        self.src_topic   = rospy.get_param("~points_topic", "/ouster/points_map")
        self.pub_pts     = rospy.Publisher("~points",  PointCloud2,    queue_size=1)
        self.pub_markers = rospy.Publisher("~markers", MarkerArray,    queue_size=1)

        # ---- ROI & 전방 제한 ----
        self.z_low       = float(rospy.get_param("~z_low", 0.02))
        self.z_high      = float(rospy.get_param("~z_high", 0.25))
        self.only_front  = bool(rospy.get_param("~only_front", True))
        self.front_axis  = rospy.get_param("~front_axis", "x")   # 'x' or 'y'
        self.front_min   = float(rospy.get_param("~front_min_m", 0.30))

        # ---- 강도 정규화 & 스레숄드 ----
        self.gamma       = float(rospy.get_param("~range_comp_gamma", 1.8))  # Ic = I * r^gamma
        self.keep_pct    = float(rospy.get_param("~keep_percentile", 99.2))  # 상위 퍼센타일
        self.min_Ic      = float(rospy.get_param("~min_Ic", 5.0))            # 절대 하한
        self.per_ring    = bool(rospy.get_param("~per_ring_normalize", True))
        self.fixed_thr   = rospy.get_param("~fixed_threshold", None)         # None 또는 float

        # ---- 클러스터 & 부스팅 ----
        self.voxel       = float(rospy.get_param("~voxel_m", 0.05))
        self.min_pts     = int(rospy.get_param("~min_cluster_pts", 6))
        self.boost       = bool(rospy.get_param("~boost", True))
        self.boost_len   = float(rospy.get_param("~boost_len_m", 0.9))
        self.boost_step  = float(rospy.get_param("~boost_step_m", 0.03))

        rospy.Subscriber(self.src_topic, PointCloud2, self.cb, queue_size=1, buff_size=2**26)
        rospy.loginfo("[tape] subscribe=%s  z=[%.2f,%.2f]  pct=%.2f  gamma=%.2f  boost=%s",
                      self.src_topic, self.z_low, self.z_high, self.keep_pct, self.gamma, self.boost)

    def cb(self, msg: PointCloud2):
        fields = {f.name for f in msg.fields}
        if   "reflectivity" in fields: inten_key = "reflectivity"
        elif "intensity"    in fields: inten_key = "intensity"
        else:
            rospy.logwarn_throttle(2.0, "[tape] no 'reflectivity' or 'intensity' field"); return
        ring_key = "ring" if "ring" in fields else None

        xs, ys, zs, I_raw, rings = [], [], [], [], []
        # 읽기
        names = ("x","y","z",inten_key, ring_key) if ring_key else ("x","y","z",inten_key)
        for p in pc2.read_points(msg, field_names=names, skip_nans=True):
            if ring_key:
                x,y,z,I,rg = p
            else:
                x,y,z,I = p; rg=-1
            if z < self.z_low or z > self.z_high: continue
            if self.only_front:
                v = x if self.front_axis=="x" else y
                if v < self.front_min: continue
            xs.append(x); ys.append(y); zs.append(z); I_raw.append(float(I)); rings.append(int(rg))

        N = len(xs)
        if N == 0:
            self._publish([], [], msg.header.stamp); return

        x = np.asarray(xs); y = np.asarray(ys); z = np.asarray(zs)
        I = np.asarray(I_raw, dtype=np.float32)
        r = np.sqrt(x*x + y*y + z*z)
        Ic = I * np.power(np.clip(r, 0.5, 200.0), self.gamma)   # 거리 보정

        if ring_key and self.per_ring:
            rings_np = np.asarray(rings, dtype=np.int32)
            Ic = self._ring_normalize(Ic, rings_np)

        thr = float(self.fixed_thr) if self.fixed_thr is not None \
              else max(np.percentile(Ic, self.keep_pct), self.min_Ic)
        keep = Ic >= thr
        K = int(keep.sum())

        rospy.loginfo_throttle(1.0, "[tape] ROI_N=%d keep=%d (thr=%.3f)", N, K, thr)

        if K == 0:
            self._publish([], [], msg.header.stamp); return

        # --- 간단 클러스터(8방향 이웃을 갖는 voxel grid) ---
        vx = self.voxel
        gx = np.floor(x[keep]/vx).astype(np.int32)
        gy = np.floor(y[keep]/vx).astype(np.int32)
        kept_idx = np.nonzero(keep)[0]
        cell2idx = defaultdict(list)
        for i,(cx,cy) in enumerate(zip(gx,gy)): cell2idx[(cx,cy)].append(kept_idx[i])

        visited=set(); clusters=[]
        neigh=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
        for cell in list(cell2idx.keys()):
            if cell in visited: continue
            q=deque([cell]); visited.add(cell); cur=[]
            while q:
                c=q.popleft(); cur+=cell2idx[c]
                cx,cy=c
                for dx,dy in neigh:
                    n=(cx+dx,cy+dy)
                    if n in cell2idx and n not in visited:
                        visited.add(n); q.append(n)
            if len(cur)>=self.min_pts: clusters.append(np.asarray(cur, dtype=np.int32))

        # --- 결과 포인트 & 디버그 마커 ---
        out_points = np.stack([x[keep], y[keep], z[keep]], axis=1).tolist()
        markers = MarkerArray(); mid=0

        if self.boost:
            # 클러스터별 PCA 주성분 축으로 얇은 포인트를 추가해 선 검출을 돕는다
            for idx in clusters:
                X = np.stack([x[idx], y[idx]], axis=1)
                C = np.cov(X.T) + 1e-6*np.eye(2)
                w,v = np.linalg.eigh(C)  # v[:,1]이 주성분
                axis = v[:,1] / (np.linalg.norm(v[:,1])+1e-9)
                ctr = X.mean(axis=0)

                half = 0.5*self.boost_len
                steps = int(max(2, round(self.boost_len/self.boost_step)))
                for t in np.linspace(-half, half, steps):
                    px,py = ctr + t*axis
                    out_points.append([float(px), float(py), float(self.z_low)])

                # 디버그: 클러스터 박스 & 축
                markers.markers.append(self._mk_box(mid, ctr, w, self.z_low, msg.header.stamp)); mid+=1
                markers.markers.append(self._mk_axis(mid, ctr, axis, self.boost_len, self.z_low, msg.header.stamp)); mid+=1
        else:
            # 디버그: 원시 클러스터만
            for idx in clusters:
                X = np.stack([x[idx], y[idx]], axis=1)
                C = np.cov(X.T) + 1e-6*np.eye(2)
                w,v = np.linalg.eigh(C); axis=v[:,1]/(np.linalg.norm(v[:,1])+1e-9)
                ctr=X.mean(axis=0)
                markers.markers.append(self._mk_box(mid, ctr, w, self.z_low, msg.header.stamp)); mid+=1
                markers.markers.append(self._mk_axis(mid, ctr, axis, 0.6, self.z_low, msg.header.stamp)); mid+=1

        self._publish(out_points, markers, msg.header.stamp)

    # ---------- helpers ----------
    def _ring_normalize(self, Ic, rings):
        out = np.empty_like(Ic)
        for rg in np.unique(rings):
            m = (rings==rg)
            if m.sum()<10: out[m]=Ic[m]; continue
            p50 = np.percentile(Ic[m], 50.0)
            p90 = np.percentile(Ic[m], 90.0)
            scale = (p90-p50) if (p90>p50) else 1.0
            out[m] = (Ic[m]-p50)/max(scale,1e-6)
        return out

    def _mk_box(self, mid, ctr, w, z, stamp):
        # w: eigenvalues (분산), 근사로 가로/세로를 잡자
        mk = Marker()
        mk.header.frame_id=self.frame_id; mk.header.stamp=stamp
        mk.ns="tape"; mk.id=mid; mk.type=Marker.CUBE; mk.action=Marker.ADD
        major = math.sqrt(max(w[1],1e-6))*4.0
        minor = math.sqrt(max(w[0],1e-6))*2.0
        mk.scale.x=major; mk.scale.y=minor; mk.scale.z=0.03
        mk.color=ColorRGBA(0.1,0.8,1.0,0.5)
        mk.pose.orientation.w=1.0
        mk.pose.position.x=float(ctr[0]); mk.pose.position.y=float(ctr[1]); mk.pose.position.z=float(z)
        return mk

    def _mk_axis(self, mid, ctr, axis, L, z, stamp):
        mk = Marker()
        mk.header.frame_id=self.frame_id; mk.header.stamp=stamp
        mk.ns="tape_axis"; mk.id=mid; mk.type=Marker.LINE_LIST; mk.action=Marker.ADD
        mk.scale.x=0.03; mk.color=ColorRGBA(0.1,0.9,0.2,0.9)
        p1 = ctr - 0.5*L*axis; p2 = ctr + 0.5*L*axis
        mk.points = []
        from geometry_msgs.msg import Point
        a=Point(x=float(p1[0]), y=float(p1[1]), z=float(z))
        b=Point(x=float(p2[0]), y=float(p2[1]), z=float(z))
        mk.points.extend([a,b])
        return mk

    def _publish(self, pts_list, markers, stamp):
        cloud = _cloud_xyz32(pts_list, self.frame_id, stamp)
        self.pub_pts.publish(cloud)
        if isinstance(markers, MarkerArray):
            self.pub_markers.publish(markers)
        else:
            self.pub_markers.publish(MarkerArray())

if __name__ == "__main__":
    rospy.init_node("parking_tape_filter_node")
    TapeFilterNode()
    rospy.spin()
