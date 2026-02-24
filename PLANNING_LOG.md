# Kiti Project - Daily Planning Log

**Project:** Kit Car Obstacle Recognition Algorithm
**GitHub Project:** [Kiti Roadmap](https://github.com/users/ShatilKhan/projects/4)
**Repository:** https://github.com/ShatilKhan/Kiti

---

## 2026-01-26

### Context Gathered

**Research Papers Available:**
- **Camera Motion Compensation** (8 papers)
  - Computational complexity of motion estimation
  - Compensation methods for rotary-scan cameras
  - Feature tracking and motion compensation for action recognition
  - Various journal papers on camera motion compensation techniques

- **Dynamic Motion** (7 papers)
  - Fast affine motion estimation for versatile video coding
  - Block-based embedded color image/video coding
  - Runtime adaptive energy-aware motion and disparity
  - Efficient VVC motion estimation
  - Fast gradient iterative affine motion estimation

**Current Notebooks (dev branch):**
1. `Area_Marking.ipynb` - Central region marking + YOLOv5/YOLOv8 object detection + SORT tracking + distance estimation + behavior detection
2. `Kiti_Optical_flow.ipynb` - Dense optical flow + obstacle detection + path prediction using Kalman filter + linear regression trajectory prediction

**Key Techniques Already Implemented:**
- Background subtraction (MOG2)
- Dense optical flow (Farneback)
- YOLOv5/YOLOv8 object detection
- SORT object tracking
- Kalman filter for path prediction
- Linear regression for trajectory forecasting
- Distance estimation using focal length

### MCP/Tools Research

**Recommended MCP Servers to Install:**
```bash
# PDF reader for research papers
claude mcp add pdf-reader npx @fabriqa.ai/pdf-reader-mcp

# arXiv paper search
claude mcp add arxiv uvx arxiv-mcp-server

# GitHub integration (for project management)
claude mcp add github -e GITHUB_TOKEN=<your-token> -- npx -y @modelcontextprotocol/server-github
```

Use gh CLI instead of github mcp

### Tasks for Today
- [x] Set up MCP servers for PDF reading and arXiv
- [ ] Review research papers for camera motion compensation techniques
- [x] Create GitHub project items for implementation phases (8 items created)
- [ ] Plan optical flow + area marking combination pipeline

### GitHub Project Items Created
1. Research: Camera Motion Compensation Techniques
2. Research: Dynamic Motion Estimation
3. Combine Area Marking + Optical Flow Pipeline
4. Implement Camera Motion Compensation
5. Improve Object Tracking with SORT/DeepSORT
6. Distance Estimation Calibration
7. Path Prediction Enhancement
8. Real-time Performance Optimization

### Notes
- No dedicated MCP for optical flow/video analysis exists - will work with existing Python code directly
- Consider VisionCraft MCP for CV knowledge if needed

---

## 2026-02-15

### Work Done
- Completed full review of all 15 research papers (8 CMC + 7 Dynamic Motion)
- Created comprehensive learning reference documents:
  - `papers/Camera_Motion_Compensation_Context.md` — covers all 8 CMC papers with summaries, key concepts, math foundations, and relevance to Kiti
  - `papers/Dynamic_Motion_Context.md` — covers all 7 Dynamic Motion papers with the same structure

### Key Findings from Paper Review

**Camera Motion Compensation (most relevant papers):**
- **UCMCTrack (Yi et al., AAAI 2024)** — Highest relevance. Tracks objects on the ground plane instead of image plane using camera calibration. Uses Mapped Mahalanobis Distance instead of IoU. Achieves >1000 FPS on CPU. Directly applicable to Kiti's tracking pipeline.
- **Feature Tracking & Dominant Plane (Uemura et al., BMVC 2008)** — Estimates dominant plane homography via RANSAC, subtracts it to isolate independent object motion. 4x improvement in foreground/background discrimination.
- **Moving Object Detection for Moving Camera (Yu et al., IJCAS 2019)** — Homography-based frame alignment + adaptive background model. Adaptive threshold decays with camera speed. 14.8-51.2 FPS.
- **GPU Ego-Motion Compensation (Hedborg & Johansson, ~2008)** — Key insight: use regular grid points for KLT tracking instead of Harris corners (avoids tracking moving objects). 30 FPS on GPU.

**Dynamic Motion (most relevant papers):**
- **Efficient Optical Flow Motion Detection (Huang et al., 2018)** — FlowNet2.0 + quadratic background model + temporal consistency. Directly addresses non-stationary camera scenes.
- **Fast Affine ME for VVC (Park & Kang, 2019)** — 4-parameter and 6-parameter affine models capture rotation, scaling, and shear. Useful for understanding complex object motion.
- **Energy-Aware ME/DE (Zatt et al., 2011)** — Joint motion + disparity estimation for stereo cameras. Run-time adaptive quality scaling.
- **VVC ME Hardware (Ahmad et al., 2024)** — FPGA implementation achieving 4K@120fps. Shows hardware acceleration feasibility.

### Decisions Made
- Context documents will serve as the primary learning reference for implementing improvements
- Prioritized improvements for Kiti pipeline (from paper analysis):
  1. **Priority 1:** Add homography-based frame alignment before background subtraction (Papers 3, 5)
  2. **Priority 2:** Move Kalman filter tracking to ground plane (Paper 8 - UCMCTrack)
  3. **Priority 3:** Replace IoU with Mapped Mahalanobis Distance in SORT (Paper 8)
  4. **Priority 4:** Use regular grid KLT for ego-motion estimation (Paper 6)
  5. **Priority 5:** Add adaptive background model threshold based on vehicle speed (Paper 5)
  6. **Priority 6:** Implement Uniform CMC — constant params per sequence (Paper 8)
  7. **Priority 7:** Add lens distortion correction (Paper 6)

### GitHub Project Updates
- Items 1 & 2 (Research: CMC Techniques / Dynamic Motion Estimation) can be marked as complete
- Next focus should be Item 3 (Combine Area Marking + Optical Flow Pipeline) and Item 4 (Implement Camera Motion Compensation)

### Next Steps
- Begin implementing homography-based CMC into the optical flow pipeline
- Combine Area_Marking.ipynb and Kiti_Optical_flow.ipynb into a unified pipeline
- Prototype ground-plane tracking following UCMCTrack approach
- Test CMC on videos in the `videos/` folder

### Blockers/Questions
- Need to determine camera intrinsic/extrinsic parameters for ground-plane projection (may need calibration step)
- Should we start with a simple homography-only CMC or go directly to the UCMCTrack approach?

---

## 2026-02-24

### Work Done

**1. Architecture Analysis & Pipeline Unification**
- Analyzed both notebooks (`Area_Marking.ipynb`, `Kiti_Optical_flow.ipynb`) to identify overlapping code and complementary features
- Created unified pipeline: `kiti_pipeline.py` — single modular Python script combining all functionality
- Adapted from Google Colab to local execution with `videos/` directory
- Pipeline runs standalone via CLI with flexible arguments

**2. Unified Pipeline Capabilities (kiti_pipeline.py)**
- **Camera Motion Compensation**: Regular grid KLT tracking + RANSAC homography estimation + residual flow computation
- **Dense Optical Flow**: Farneback with ego-motion subtraction
- **Background Subtraction**: MOG2 on homography-compensated frames
- **Object Detection**: YOLOv8n
- **Object Tracking**: SORT with Kalman filter
- **Distance Estimation**: Focal-length based monocular depth
- **Trajectory Prediction**: Kalman filter + linear regression future path
- **8-Direction Movement Analysis**: Compass-based direction detection
- **ROI Marking**: Central 40% region with behavior detection
- **Annotation & Logging**: Full overlay + CSV/JSON export + heatmap + performance plots

**3. Literature Search — Latest Papers (2023-2026)**
Identified 15+ new papers relevant to the pipeline improvements:

| Paper | Key Contribution | Impact |
|-------|-----------------|--------|
| **EMAP** (arXiv 2404.03110) | Plug-and-play ego-motion-aware KF for SORT — 73% fewer ID switches | Highest priority |
| **MCTrack** (IROS 2025) | Decoupled KF on BEV plane, SOTA on KITTI/nuScenes/Waymo | Ground-plane tracking |
| **FoELS** (arXiv 2507.13628) | Focus of Expansion for ego-motion separation, works with Farneback | Lightweight CMC alternative |
| **BoostTrack** (MVA 2024) | Augments IoU with Mahalanobis + shape similarity | Incremental IoU improvement |
| **MONA** (arXiv 2501.13183) | YOLO + SAM + RAFT for dynamic point extraction | Modern MOG2 replacement |
| **BEVTrack** (IJCAI 2025) | Ground-plane tracking at 200 FPS | Fast BEV tracking |
| **Homography+YOLOv8** (IEEE 2024) | IPT + background subtraction, 92-99% accuracy | Validates our approach |

**4. Testing**
- Tested on `2025-05-18 17-03-38.mkv`: 30 frames, 6 tracks (car, person), 98% ROI coverage
- Tested on `2025-05-18 17-07-34.mp4`: 300 frames, 91 tracks (person, horse, kite), 55% ROI coverage
- Steady ~2.5 fps at 1920x1080 (CPU-bound, CUDA available for GPU acceleration)

### Decisions Made

1. **Unified pipeline as Python script** (not notebook) — better for version control, modularity, and CLI usage. Original Colab notebooks preserved as reference.
2. **CMC approach**: Started with homography-based (grid KLT + RANSAC) — the simplest effective approach from the literature. This answers the previous blocker question.
3. **Research-informed priority update** based on new literature:
   - **Immediate**: Integrate EMAP ego-motion-aware KF into SORT (plug-and-play, highest ROI)
   - **Short-term**: Move to ground-plane tracking (MCTrack/UCMCTrack approach)
   - **Medium-term**: Replace MOG2 with residual flow model (FoELS approach)

### GitHub Project Updates
- Item 3 (Combine Area Marking + Optical Flow Pipeline): **DONE** — `kiti_pipeline.py`
- Item 4 (Implement Camera Motion Compensation): **DONE** — CameraMotionCompensator class with grid KLT + RANSAC
- Items 1 & 2 (Research): Already complete from previous session, now updated with 15+ new papers

### Next Steps
1. **EMAP Integration**: Implement ego-motion-aware Kalman filter from EMAP paper as drop-in replacement for SORT's prediction module
2. **Ground-Plane Tracking**: Implement UCMCTrack-style ground-plane projection with camera parameter estimation
3. **Mapped Mahalanobis Distance**: Replace IoU association with MMD from UCMCTrack
4. **GPU Acceleration**: Move YOLO inference to GPU, profile bottlenecks for real-time target
5. **Full Video Processing**: Run pipeline on all 4 test videos, evaluate detection quality
6. **Adaptive Background**: Implement speed-dependent threshold from Yu et al. (IJCAS 2019)

### Blockers/Questions
- Camera intrinsic parameters still unknown — may need calibration step or use UCMCTrack's auto-estimation tool
- Current fps (~2.5) is CPU-bound. Need to profile: is the bottleneck YOLO, optical flow, or CMC?
- Should we prioritize EMAP integration (tracking improvement) or GPU acceleration (speed improvement) first?

---

## Template for Future Entries

```markdown
## YYYY-MM-DD

### Work Done
-

### Decisions Made
-

### GitHub Project Updates
-

### Next Steps
-

### Blockers/Questions
-
```
