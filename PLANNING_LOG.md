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
