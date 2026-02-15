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
