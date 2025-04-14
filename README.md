# Unified Geometric Contact Modeling

This repository provides an implementation of a unified geometric modeling framework for contact perception and interaction. The framework includes three core modules:

- **Unified Model Display**
- **Real-time Display**
- **Model Generator**

The model supports real-time contact perception, multi-scale geometric representation, and trajectory generation under controlled contact conditions.

📽️ A demonstration video (`DisplayVideo.mp4`) is included in this repository, showcasing all three modules in action.

---

## 🔧 Features

### 1. Unified Model Display
Demonstrates **uniform-speed trajectory generation** along object boundaries under a specified contact rate.  
A virtual hand of varying scale moves along the contour while maintaining consistent contact, based on the proposed unified model.

### 2. Real-time Display
Illustrates the **real-time perception capabilities** of the unified model.  
Given arbitrary hand positions (e.g., simulated via mouse), the model instantly infers and updates the contact state.

### 3. Model Generator
Generates contact-aware geometric structures for a given target region, including:
- `E(A,B)`: External boundary  
- `I(A,B)`: Internal boundary  
- `P(A,B)`: Contact region  

Inputs:
- A binary image (target region)  
- A scale parameter  
- A contact resolution parameter  

The output is a structured geometric representation based on the unified model.

---

## 📦 Dependencies

- OpenCV `v3.4.5` – Used for image reading, writing, and display.

---

## 🛠 Build Instructions

The core model functionality is provided via a compiled static and dynamic library:

- Static Library: `UnifiedModel.lib`  
- Dynamic Library: `UnifiedModel.dll`  
- Header File: `interface.h`  

### Integration Steps

1. Include `interface.h` in your project.
2. Link against `UnifiedModel.lib`.
3. Place `UnifiedModel.dll` in your executable directory or system path.
4. Ensure OpenCV 3.4.5 is correctly linked and configured.

---

## 📄 License

This project is intended for academic and research use only.  
For commercial licensing, please contact the authors.

---

## 📫 Contact

For questions, feedback, or collaboration, feel free to open an issue or reach out via email.
