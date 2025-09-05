# Uncovering Behavioral Hierarchies in DVC Mouse Data with hBehaveMAE

## 1. Overview
This repository outlines the conceptual framework for applying the **Hierarchical Masked Autoencoder for Behavior (hBehaveMAE)** to time-series data from the **Digital Ventilated Cage (DVC) system**. The goal is to move beyond simple activity counts and learn a rich, multi-layered representation of mouse behavior in a self-supervised manner.  

By using hBehaveMAE, we can analyze activity patterns at multiple scales—from brief, minute-long movements to complex, multi-day rhythms—without requiring manually labeled behaviors.

---

## 2. The DVC Dataset: A Spatio-Temporal Challenge 🐭
The DVC system provides a unique type of behavioral data that is perfectly suited for spatio-temporal modeling. Understanding its structure is key to applying hBehaveMAE effectively.

### Spatial Component: The 4x3 Electrode Grid
At any given moment, the mouse's location and activity are captured by 12 capacitive electrodes arranged in a fixed 4x3 grid under the cage floor. This grid represents the spatial dimension.  

**Why preserve the 4x3 structure?**  
Treating electrodes as a simple list of 12 numbers discards spatial relationships. Maintaining the grid allows the model to learn meaningful patterns, such as:
- Movement from the front to the back of the cage.
- Dwelling time in a specific corner.
- Differences between activity near the food hopper versus the water bottle.

### Temporal Component: Continuous Minute-by-Minute Record
The DVC system records an activation value for each electrode once per minute. This continuous stream forms the temporal dimension and allows analysis of:
- When a mouse is active.
- How activity unfolds over time.
- Multi-scale behavioral patterns.

---

## 3. The hBehaveMAE Model: Learning the Structure of Behavior
**hBehaveMAE** is a self-supervised learning framework based on the **Transformer architecture**, designed for spatio-temporal data like pose trajectories. Its key strength is learning a **hierarchical representation of behavior**, capturing multiple temporal scales.

### Hierarchical Embeddings
- **Low-Level Embeddings:** Capture short-term events (e.g., a few minutes of movement).  
- **Mid-Level Embeddings:** Represent larger behavioral "chunks" (e.g., transition from rest to activity).  
- **High-Level Embeddings:** Summarize long-term patterns (e.g., diurnal rhythms over 24 hours or week-to-week variations).

---

## 4. The DVCDataset Class: Bridging Data and Model
The custom `DVCDataset` PyTorch class connects raw DVC data to the hBehaveMAE model.

### Responsibilities
1. **Loading Data:** Reads raw summary files containing 12-column electrode data aggregated over a period (e.g., daily).  
2. **Enforcing Spatial Structure:** Reshapes the linear 12-electrode sequence into the 4x3 grid.  
3. **Preparing Sequences:** Prepares data into subsequences of a fixed length (`num_frames`) for the model.

---

## 5. Designing a Hierarchy with Training Parameters
We can control the hierarchy the model learns through training parameters:

- **`patch_kernel`:** Defines the smallest spatio-temporal "patch" or token.  
  Example: `(15, 1, 1)` → each token represents 15 minutes of single-electrode activity.  

- **`stages` & `q_strides`:** Guide the progressive merging of tokens.  
  Example: `q_strides = 4,1,1; 3,1,1; ...`  
  - First stage: merge 4 temporal tokens → 15-min tokens become 60-min representations.  
  - Second stage: merge 3 temporal tokens → 60-min → 3-hour representations.  

By tuning these, we can generate embeddings for desired time scales (e.g., 15 min, 1 hr, 3 hr, 6 hr, 12 hr, 1 day).

---

## 6. Extracting and Interpreting Hierarchical Embeddings
After training, the embeddings can be extracted for analysis.

### What is an Embedding?
An **embedding** is a dense numerical vector summarizing behavior within a specific spatio-temporal window.  

### How Spatial Information is Included
- Final embeddings summarize spatial information rather than preserving full spatial maps.  
- Output files are separated by temporal level (e.g., `level1_15min`, `level2_60min`).  
- Pooling mechanisms average over spatial tokens to produce a holistic feature vector.  
- The model reconstructs the original 4x3 grid, packing distinguishing spatial features into embeddings.

**Example:**  
- One pattern may represent high activity in the front of the cage.  
- Another pattern may represent activity in the back.  

This allows analysis of overall spatial states without full spatial maps for every timepoint.

---
