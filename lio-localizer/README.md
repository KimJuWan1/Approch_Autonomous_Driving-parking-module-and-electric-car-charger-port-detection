# Lio Localizer

## 1. Installation

### 1-2. nanoflann

```
cd ~/git
git clone https://github.com/jlblancoc/nanoflann.git
cd nanoflann
git checkout tags/v1.4.3
mkdir build && cd build && cmake ..
make
sudo make install
```

### 1-3. ceres

```
sudo apt-get install cmake libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev

cd ~/git
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
git checkout tags/2.1.0
mkdir build && cd build && cmake ..
make -j3
sudo make install
```

### Play
```
# robot
roslaunch lio_localizer run-robot.launch
rosbag play 2023-12-01-19-52-02.bag

# Kitti (patial map in 40 seconds)
# In order to initialize the pose using GICP, you have to pause the replay, because the GICP is slow.
roslaunch lio_localizer run-kitti.launch
rosbag play 2011_09_30_drive_0028.bag -s 2
```
