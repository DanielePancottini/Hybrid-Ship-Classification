# WaterScenes Transformer-Based Architecture for Ship Detection in Radar Imagery

## Project Aim & Abstract

**Goal:**
The initial objective was to design a multi-modal object detection system fusing **4D Radar** and **RGB Camera** data for maritime environments. However, due to significant computational constraints (specifically limited VRAM), the project scope was adjusted to focus exclusively on the radar branch. The final implementation develops a **Radar-only detection network** using a hybrid architecture. By combining **Convolutional Neural Networks (CNNs)** with **Transformer Encoders**, the model aims to enhance local feature extraction while recovering long-range dependencies often lost in sparse radar point clouds.

**Motivation:**
Object detection in maritime settings is difficult for standard optical cameras. Water reflections, lack of surface texture, and adverse weather (fog, rain) often lead to failure. Millimeter-wave (mmWave) Radar is a strong alternative because it provides reliable depth and velocity data regardless of visibility. The main challenge is that radar data is sparse and lacks semantic resolution compared to images. This project explores if a more complex architecture (RCNet with Transformers) can compensate for the absence of RGB data.

**Selected Papers & Benchmark:**
The project utilizes the **WaterScenes** [3] benchmark for dataset and evaluation protocols. The architectural choices were inspired by **Achelous++** [1] and **WS-DETR** [2], particularly their approaches to processing non-homogeneous maritime data and utilizing attention mechanisms for feature enhancement.

**References:**
1.  **Achelous++:** *Water-surface Object Detection based on Fusion of Camera and 4D Radar.* (arXiv:2312.08851)
2.  **WS-DETR:** *Multi-modal Maritime Object Detection with Transformers.* (arXiv:2504.07441v1)
3.  **WaterScenes:** *A Multi-Task 4D Radar-Camera Fusion Dataset and Benchmark.* (arXiv:2307.06505)
