# SLAM_practice

![C++](https://img.shields.io/badge/C++-00599C?style=flat&logo=c%2B%2B&logoColor=white)
![ROS2](https://img.shields.io/badge/ROS2-Humble-blue?style=flat&logo=ros)
![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-E95420?style=flat&logo=ubuntu&logoColor=white)

이 프로젝트는 ORB 기반 Visual SLAM 시스템을 구현해본 프로젝트이다.  
전통적인 수치 최적화 방식의 Frontend를 구현하고, 더 나아가 딥러닝 기반의 LoFTR를 Triangulation 단계에 추가하여 특징이 부족한 환경에서도 풍부한 맵포인트를 생성할 수 있도록 구현하였다.

---


## System Architecture

시스템은 크게 트래킹을 담당하는 Frontend와 최적화를 담당하는 Backend로 구성된다.

### 1. Frontend (Visual Odometry)
* **Feature Matching:** OpenCV의 ORB Detector와 KNN 매칭 방식.
* **Pose Estimation:** 최초의 포즈 추정은 2D-2D 매칭(F, H, E Matrix), 이후에는 3D-2D 매칭(PnP)을 통해 카메라 포즈를 추정.
* **Window-based Search:** LK Optical Flow로 예측된 좌표 주변의 윈도우 또는 맵 포인트를 Image plane에 투영한 좌표 주변의 윈도우 내에서만 디스크립터 매칭을 수행하여 연산 효율 향상.



### 2. Triangulation (LoFTR Integration)
* **Motivation:** 실시간 Tracking에 LoFTR(RTX5060 laptop 기준 ~350ms)을 직접 쓰기엔 제약 존재.
* **Solution:** Tracking은 가벼운 ORB를 유지하고, 새로운 키프레임 등록 시 **삼각측량(Triangulation)** 단계에만 LoFTR을 적용하여 정밀한 맵 포인트를 생성. 텍스쳐가 없는 이미지에서도 더 많은 맵 포인트를 생성하고, 더 정확한 매칭을 할 수 있을 것으로 생각.



### 3. Backend (Local BA)
* **Local BA:** 새로운 키프레임 추가 시, 이전 **5개의 키프레임**과 이를 공통으로 관측하는 맵 포인트들을 최적화하여 오차 누적을 방지.
* **Optimizer:** g2o 라이브러리를 사용하며, 이상치 제거를 위해 Huber Robust Kernel을 적용.

---

## 📐 Mathematical Background

### Epipolar Geometry & Pose Estimation
두 뷰 사이의 기하학적 관계는 다음과 같은 에피폴라 제약 조건을 만족.  
```math
x'^T E x = 0, \quad E = R[t]_{\times}
```
본 시스템은 $F, H$ 스코어를 비교하여 충분한 시차(Parallax)가 확보된 경우에만 지도를 초기화.

### PnP
맵 포인트의 3D 좌표는 고정돼있고, 해당 맵 포인트를 투영한 좌표와 매칭되는 2D 포인트의 좌표 오차합을 최소화하여 카메라의 Pose를 최적화.


### Local Bundle Adjustment
재투영 오차(Reprojection Error)를 최소화하는 카메라 자세($T_k$)와 맵 포인트($X_p$)를 찾는다.  
```math
\min_{\{T_k\}, \{X_p\}} \sum_{k \in \mathcal{K}_L} \sum_{p \in \mathcal{P}_L} \rho \left( \| z_{kp} - \pi(K, T_k, X_p) \|^2_{\Sigma} \right)
```


---

## 📊 Results

* **Dataset:** EuRoC MAV (Vicon Room 1 01 easy).
* **Analysis:** 특징점이 많은 텍스처(체커보드 등) 환경에서 안정적인 트래킹을 확인했습니다.

---

## 🛠 Prerequisites

* **ROS2 Humble**
* **OpenCV 4.5.4**
* **Eigen 3.4.0**
* **g2o (Graph Optimization Library) 2020.5.29**
* **PyTorch (for LoFTR Inference node) 2.9.1+cu128**

---

## 🗺 Roadmap
- [x] Frontend: ORB-based Tracking
- [x] Triangulation & Map Point generation
- [x] Backend: g2o-based Local BA
- [x] LoFTR Hybrid Matching 
- [ ] Global Bundle Adjustment (Implementation planned) 
- [ ] Loop Closure (Implementation planned)

---
