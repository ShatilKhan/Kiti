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

## 2026-02-24 (Session 2 — Paper & Documentation)

### Work Done

**1. Research Paper Draft (`docs/paper.md`)**
- Wrote full structured research paper (422 lines, 38KB):
  - Abstract, Introduction (3 contributions)
  - Related Work: 4 subsections (Detection, Tracking, CMC, Distance/Trajectory)
  - Proposed Method: 9 subsections (System Overview through ROI/Behavior)
  - 2 pseudocode algorithms in ByteTrack/UCMCTrack convention
  - Experiments: 6 subsections with tables
  - Comparison table: SORT vs DeepSORT vs ByteTrack vs BoT-SORT vs UCMCTrack vs Kiti
  - Discussion: Current Limitations + Planned Improvements
  - Conclusion + 24 references

**2. Formatted Word Document (`docs/paper.docx`)**
- Created `docs/generate_paper_docx.js` — reusable Node.js docx-js script (59KB)
- Formatting: Times New Roman 12pt, US Letter, 1" margins, numbered sections
- Title page: Title (18pt bold), author "Shatil Khan", date "February 2026"
- Header/footer: Paper title (right, italic) + page numbers (centered)
- Pseudocode blocks: Courier New 9pt with horizontal rule borders
- Tables: Black borders, blue header shading, cell padding
- Figure placeholders: Dashed gray border, italic description text
- References: Hanging indent, italic venue names
- Validated with anthropics-docx schema validator (all checks passed)
- Fixed known docx-js border ordering issue (top/left/bottom/right schema order)

**3. Additional Conversions**
- `docs/paper.pdf` (83KB) via weasyprint (math rendering approximate)
- `docs/paper.html` (69KB) intermediate format

**4. Pipeline Architecture Diagram (Figure 1)**
- Created `docs/figures/generate_pipeline_diagram.py` — reusable Python/matplotlib script
- Generated `docs/figures/pipeline_architecture.png` (350KB, 300dpi) — publication-quality 10-stage flow diagram
- Generated `docs/figures/pipeline_architecture.pdf` (45KB) — vector version
- Also created `docs/figures/pipeline_architecture.tex` — LaTeX/TikZ source (needs texlive-pictures to compile, kept for future use)
- Diagram features: two-column layout, numbered color-coded stages, subtitles with method details, data flow arrows, dashed red H matrix shared parameter line, legend
- LaTeX compilation not available (only texlive-base installed, missing texlive-pictures for TikZ), so used matplotlib instead

**5. Documentation & Memory**
- Created `memory/docx_workflow.md` with full regeneration instructions
- Updated project memory with file structure, docx workflow, daily update process
- All scripts kept in codebase for daily reuse:
  - `docs/generate_paper_docx.js` — regenerate Word document
  - `docs/figures/generate_pipeline_diagram.py` — regenerate pipeline diagram

### Decisions Made
1. **Reusable generator scripts** — all kept in repo for daily reuse
2. **Regeneration commands**:
   - DOCX: `NODE_PATH=$(npm root -g) node docs/generate_paper_docx.js`
   - Pipeline diagram: `python3 docs/figures/generate_pipeline_diagram.py`
3. **Daily workflow**: Edit paper.md + update generator → regenerate docx → share
4. **Figure placeholders** left as tagged text for manual image insertion
5. **Matplotlib over LaTeX** for diagram generation — texlive-pictures not installed, matplotlib available everywhere

### Image Placeholders in paper.docx

| # | Placeholder Description | Target Path | Status |
|---|------------------------|-------------|--------|
| 1 | Pipeline architecture diagram (10-stage flow) | `docs/figures/pipeline_architecture.png` | **CREATED** (350KB, 300dpi) |
| 2 | Sample annotated frame (bboxes, IDs, distances, ROI) | Extract frame from `output/2026-02-24/2025-05-18 17-07-34_annotated.mp4` | **NEEDS EXTRACTION** (use ffmpeg) |
| 3 | CMC comparison heatmap (with vs without CMC) | `docs/figures/cmc_comparison_heatmap.png` | **NEEDS CREATION** (run pipeline with --no-cmc, compare) |
| 4 | Processing time per frame plot | `output/2026-02-24/2025-05-18 17-07-34_performance.png` | **EXISTS** — ready to paste |
| 5 | Motion heatmap from Seq-1 | `output/2026-02-24/2025-05-18 17-07-34_heatmap.png` | **EXISTS** — ready to paste |

### Next Steps
1. ~~Generate pipeline flow diagram~~ — **DONE** (`docs/figures/pipeline_architecture.png`)
2. **Extract sample annotated frame** from output video for Figure 2 (use ffmpeg)
3. **Generate CMC comparison** (run pipeline with/without --no-cmc) for Figure 3
4. **Paste existing images** into docx: performance.png (Fig 4), heatmap.png (Fig 5)
5. Begin EMAP integration into tracking module
6. Run pipeline on all 4 test videos for full results section

### Blockers/Questions
- CMC comparison requires two pipeline runs (with and without --no-cmc flag)
- texlive-pictures not installed — used matplotlib for diagrams instead of TikZ

---

## 2026-02-28

### Work Done

**1. GitHub Project Roadmap Update**
- Marked 4 items as Done: Research CMC, Research Dynamic Motion, Combine Pipeline, Implement CMC
- Added 8 new items: Algorithm Flow Chart (Done), Pseudo Code (Done), Research Paper Draft (In Progress), Road Condition Recognition, Traffic Sign Recognition, Road Slope Detection, EMAP Integration, Module Fusion
- Total: 16 items (6 Done, 1 In Progress, 9 Todo)

**2. CMC Comparison Experiment (Figure 3)**
- Ran pipeline twice on Seq-1 (300 frames): with CMC and without CMC
- Generated side-by-side heatmap comparison at `docs/figures/cmc_comparison_heatmap.png`
- Quantitative results: RANSAC inlier ratio mean=0.983 (σ=0.034), min=0.795, max=1.000
- Detection/tracking metrics identical (CMC affects flow/BG subtraction, not YOLO/SORT)
- 92% of frames had inlier ratios above 0.95

**3. Annotated Frame Extraction (Figure 2)**
- Extracted frame 150 from annotated video to `docs/figures/sample_annotated_frame.png`
- Shows bounding boxes, track IDs, distance estimates, and ROI overlay

**4. Research Paper Enhancement (paper.md: 422 → 446 lines)**
- Based on comprehensive literature review (15+ new papers found):
  - Fixed factual error: YOLOv8 no longer claimed as "latest" — added YOLOv10, YOLO26
  - Corrected SORT state vector description (7D, not 4D)
  - Added 6 new methods to Related Work: Deep OC-SORT, BoostTrack++, HybridTrack, PD-SORT, Depth Anything V2, SEA-RAFT
  - Expanded comparison table with 3 new methods + FPS measurement caveat footnote
  - Rewrote CMC analysis section with quantitative inlier ratio data
  - Added hardware specs (AMD Ryzen, CUDA 12.8, PyTorch 2.10)
  - Justified OR mask combination strategy (complementary failure modes)
  - Improved scientific writing quality (removed informal language)
  - Added 9 new references (total: 33)
  - Concrete upgrade paths in Discussion: SEA-RAFT, YOLO26, Depth Anything V2, Metric3D v2

**5. Key Methodological Gaps Identified (for future work)**
- No standard MOT metrics (HOTA, MOTA, IDF1) — need ground truth annotations
- No distance estimation validation against reference measurements
- No ablation studies (grid spacing, motion threshold, mask combination strategy)
- Results shown for only 300 frames of 1 sequence — need full evaluation on all 4 sequences
- Trajectory prediction not evaluated (no ADE/FDE metrics)
- Planar assumption failure cases not analyzed

### Decisions Made
1. CMC ablation shows identical detection/tracking (by design — CMC only affects flow/BG subtraction)
2. Paper needs quantitative ground truth evaluation before submission — flagged as next priority
3. More pseudocode will be added as new features are implemented (per README features list)
4. User already added images to docx manually — generator script serves as clean rebuild template

### GitHub Project Updates
- 6 items marked Done, 1 In Progress, 9 Todo
- New items track all README features + research priorities

### Image Placeholders Status (all figures now exist)

| # | Description | Path | Status |
|---|-------------|------|--------|
| 1 | Pipeline architecture diagram | `docs/figures/pipeline_architecture.png` | **CREATED** |
| 2 | Sample annotated frame | `docs/figures/sample_annotated_frame.png` | **CREATED** |
| 3 | CMC comparison heatmap | `docs/figures/cmc_comparison_heatmap.png` | **CREATED** |
| 4 | Processing time plot | `output/2026-02-28/with_cmc/..._performance.png` | **EXISTS** |
| 5 | Motion heatmap | `output/2026-02-28/with_cmc/..._heatmap.png` | **EXISTS** |

### Next Steps
1. Run pipeline on all 4 test sequences for full experimental results
2. Create ground truth annotations for at least 1 sequence (for MOT metrics)
3. Add ablation studies: grid spacing, motion threshold, YOLO confidence
4. Begin EMAP integration (highest priority pipeline improvement)
5. Add pseudocode for future features: traffic sign recognition, road condition, slope detection
6. Update docx with all new figures and content

### Blockers/Questions
- Ground truth annotation tool needed (CVAT or similar) for quantitative evaluation
- Should ablation studies prioritize CMC parameters or tracking parameters?

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
