# MODVORTEx

**Magneto-Optical Domain Velocity Observation and Real-Time Extraction**

Version 1.3.5 | Author: Rakhul Raj

---

## Table of Contents

1. [Overview](#overview)
2. [Theory](#theory)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [GUI Reference](#gui-reference)
   - [Main Window Layout](#main-window-layout)
   - [Image Tabs](#image-tabs)
   - [Folder Selection](#folder-selection)
   - [Measurement Type](#measurement-type)
   - [Binarization Controls](#binarization-controls)
   - [Direction Selection](#direction-selection)
   - [Circular Expansion](#circular-expansion)
   - [Domain Center Selection](#domain-center-selection)
   - [Calculation & Export](#calculation--export)
   - [Menu Bar](#menu-bar)
6. [Measurement Workflows](#measurement-workflows)
   - [Bubble Directional Workflow](#bubble-directional-workflow)
   - [Arbitrary Structure Workflow](#arbitrary-structure-workflow)
    - [Bubble Circle Fit Workflow](#bubble-circle-fit-workflow)
    - [Manual Workflow](#manual-workflow-no-coded-pulse-widths-needed)
7. [Settings Window](#settings-window)
8. [Image Viewer](#image-viewer)
9. [REST API](#rest-api)
10. [File Format & Data Export](#file-format--data-export)
11. [Troubleshooting](#troubleshooting)
12. [Citing MODVORTEx](#citing-modvortex)

---

## Overview

MODVORTEx is a desktop application for analyzing **magnetic domain wall motion** from sequences of microscopy images. It is designed for researchers studying **field- or current-driven domain wall dynamics** in magnetic thin films and nanostructures.

The software:

- Loads a series of magnetic domain microscopy images (PNG, JPEG, WebP) acquired after magnetic pulses of varying pulse width.
- Tracks domain wall displacement across the image sequence using one of three measurement methods.
- Outputs displacement-vs-pulse-width curves used for calculating domain wall velocity.
- Batch-processes all pulse-amplitude (voltage/field) folders in a dataset using multiprocessing.
- Provides a REST API for remote control and automation.

---

## Theory

When a magnetic field or current pulse is applied to a magnetic material with perpendicular magnetic anisotropy, **magnetic domains** (regions of uniform magnetization) expand or contract via **domain wall motion**. By capturing microscopy images of the domain structure after pulses of different widths (and amplitudes), the displacement of the domain wall can be measured as a function of pulse width. The **domain wall velocity** is extracted from this displacement curve (velocity = displacement / pulse width in the steady-flow regime, or the slope of the curve more generally).

MODVORTEx supports three measurement strategies:

| Method | Description | Best For |
|---|---|---|
| **Bubble Directional** | Draws a line through a bubble domain and tracks where the domain wall crosses that line | Circular bubble domains with a known center |
| **Arbitrary Structure** | Detects the wall by edge detection and traces the full contour of any shaped domain | Non-circular, irregular domain patterns |
| **Bubble Circle Fit** | Fits a minimum enclosing circle to each bubble domain and tracks the change in radius | Circular bubble domains expanding radially |

---

## Installation

### Prerequisites

- Python 3.9 or newer (tested with Python 3.12)
- Windows, Linux, or macOS

### Install from source

```bash
# Clone or download the repository
cd MODVORTEx

# Create a virtual environment (recommended)
python -m venv venv
.\venv\Scripts\activate    # Windows
source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

> **Note:** the two helpers formerly taken from the external `mymodule` package (`decorate_all_methods`, `exception_handler`) are vendored in `extras.py`, so no extra package needs to be installed.

### Run from source

```bash
python main.py
```

### Build a standalone executable (PyInstaller)

The program loads its `.ui` files, the `settings/` directory, and `icons/` assets relative to the application path at runtime, so they must be bundled:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --add-data "icons;icons" --add-data "settings;settings" --add-data "*.ui;." main.py
```

(On Linux/Mac, replace `;` with `:` in `--add-data` arguments.)

---

## Quick Start

1. **Launch MODVORTEx** – a loading screen appears while heavy modules initialize, then the main window opens.
2. **Set up settings** – open `View > Settings` and set the **Scale** (physical distance per pixel of your imaging system) and the image format you use.
3. **Select a folder** – click `Select Folder`, pick any file inside the folder that directly contains your image sequence. That folder's path is filled into the text box.
4. **Choose measurement type** – select `BUBBLE_DIRECTIONAL`, `ARBITRARY_STRUCTURE`, or `BUBBLE_CIRCLE_FIT` from the dropdown.
5. **Load images** – click `Load` to display the image sequence in the tabs.
6. **Define measurement parameters** – set binarization options, draw the direction line or set the domain center (see workflows below).
7. **Calculate** – click `Calculate Current` to measure displacement for the loaded set, or `Calculate All` to batch-process every amplitude folder in the dataset.
8. **Export** – For `Calculate Current` the average displacement and the standard deviation is printed in the UI itself while for `Calculate All` the results are saved automatically as tab-separated `.txt` files in the `DisplacementTime` folder (configurable). Use `Plot Current` to visualize them.

---

## GUI Reference

### Main Window Layout

The window has a top part for viewing images and results, and a bottom part for settings and actions:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Menu Bar: File | View | Help                                                 │
├──────────────────────────────────────────────┬───────────────────────────────┤
│ Image tabs (top-left, large area)            │ Folder path box               │
│  img2[x] img3[x] img4[x] img5[x] edges[x]    │  e.g. C:\Users\...\4.2V_17     │
│                                              |[Select Folder] [Load]          │
│                                              │          [Start API]            │
│  Photo area: black image with white outlines,├───────────────────────────────┤
│  red measurement line with green side lines  │ Results box                   │
│  (scroll bars appear if photo is too big,   │  Displacement: 22.47            │
│   each tab can be closed with its [x],       │  Standard Deviation: 0.88       │
│   use Left / Right arrow keys to move        │  ------------------------------ │
│   between photos)                            ├───────────────┬───────────────┤
│                                              │ Measurement   │ Choice box:   │
│                                              │ Type          │ BUBBLE_CIRCLE_FIT │
│                                              ├───────────────┼───────────────┤
│                                              │Calculate      │ Get edge      │
│                                              │current        │               │
├──────────────────┬───────────────────────────┼───────┬───────┴───────┬───────┤
│ Binarization     │ Select Direction          │Circular │ Result Save │ Save  │
│                  │                           │Expansion│ Folder:     │name:  │
│ [Picture button] │ [Load] Point 1: 434, 35   │roundness:│ Overwrite if save │DisplacementTime │
│ Threshold type:  │        Point 2: 232, 37   │ 1.20    │ folder exists checkbox│               │
│  Custom / Otsu   │ Width dial + [Set]        │Select   ├───────────────┬─────┤
│ Threshold Value: │ Ticked? Set to all        │Domain:  │Bulk Calculate │Calculate All│
│  144             │ Current Coordinates       │(x,y)... │(greyed out)   │(shows "Stop" │
│ Ticked? Inverse  │ (pixels): (143, 1)        │[Auto    │               │while busy)│
│ [Plot Histogram] │ - mouse position shows here│ select]│               │             │
│ Current Threshold│                           │[Show Fit]│              │[Plot Current]│
│ Value: None      │                           │       │               │small button │
│                  │                           │       │               │bottom-right │
├──────────────────┴───────────────────────────┴───────┴───────────────┴───────────┤
│ D Center Select (small box, bottom-left): e.g. 432,29 - default is 255,255       │
└──────────────────────────────────────────────────────────────────────────────┘
```

What is where:

- **Top-left:** Your photos. Each photo gets its own tab like img2, img3, edges, domain_fit. Pressing Get edge adds a new edges tab. Pressing Show Fit adds a domain_fit tab.
- **Top-right:** Folder path box at the top, with Select Folder on the left and Load on the right below it, and Start API centered below those. Under that is the results box where Displacement and Standard Deviation are printed.
- **Middle-right:** Measurement Type choice box, with Calculate current and Get edge buttons below it side-by-side.
- **Bottom-right:** Result Save Folder name, Overwrite if save folder exists tick box, Bulk Calculate (normally greyed out) and Calculate All side-by-side, with a small Plot Current button at the very bottom-right corner.
- **Bottom (left to right):** Binarization box, then Select Direction box, then Circular Expansion box. The Current Coordinates readout lives inside the Select Direction box - it is not in a bottom status bar. D Center Select is only a small box at the bottom-left.

### Image Tabs

The large area on the left is a tab widget. Each loaded image gets its own tab labelled `img0`, `img1`, etc.

- **Navigation:** Click tabs to switch between images, or use the **Left/Right arrow keys** to step through tabs.
- **Close tabs:** Each tab has a close button (x). Click `Load` to reload all tabs from the folder.
- **Scroll:** If an image is larger than the viewport, scroll bars appear automatically (each tab is wrapped in a scroll area).
- **Drawing lines:** For the directional methods, click directly on an image to define the measurement line:
  - **Left-click** sets the first endpoint.
  - **Left-click again** sets the second endpoint (a red line is drawn).
  - **Right-click drag** moves the whole line.
  - **Left + right click together** clears the line.

### Folder Selection

**Controls at the top-right:**

- **Folder path box:** Shows the path of the folder whose images are (or will be) loaded. You can also paste a path directly into it.
- **`Select Folder` button:** Opens a file dialog. Pick any file inside the folder that directly contains your image sequence — the folder containing that file is filled into the path box.
- **`Load` button:** Reads all images of the configured format from that folder and loads them into the tabs. Tabs follow filename order with the reference image first. If the file names carry no coded pulse width (e.g. `frame1.png`), a **Confirm image order** window pops up showing the predicted order — drag rows or use Move Up / Move Down, check the preview, then press OK. Your choice is remembered for the folder; reopen it anytime via `View > Reorder Images`. Cancelling keeps the currently loaded tabs.

**Expected folder structure for batch processing:**

```
Dataset/
├── 25V_1/
│   ├── 1p1s.png
│   ├── 1p1s_01.png
│   └── ...
├── 25V_2/
│   └── ...
├── 30V_1/
│   └── ...
```

Point the path box at **one** image folder (e.g. `25V_1`). That folder is what you inspect and measure interactively; `Calculate All` then processes all of its sibling amplitude folders (`25V_1`, `25V_2`, `30V_1`, …). Each sibling folder name must start with the amplitude value followed by a non-digit character (e.g. `25V_1`), and images inside must encode the pulse width in their file names (e.g. `1p0us.png`).

### Measurement Type

**`measType` dropdown (3 options):**

| Option | Index | Description |
|---|---|---|
| `BUBBLE_DIRECTIONAL` | 0 | Draw a line through a bubble domain; measures wall displacement where the wall crosses that line. Requires a domain center. |
| `ARBITRARY_STRUCTURE` | 1 | Detects the wall by edge detection and traces the contour of arbitrary-shaped domains. Uses the direction line. |
| `BUBBLE_CIRCLE_FIT` | 2 | Fits a minimum enclosing circle to each detected bubble; measures the change in radius. No direction line needed. |

### Binarization Controls

**Group box labeled "Binarization"**

All domain wall detection starts with **binarization** – converting the grayscale microscopy image into a binary (black/white) image in which the domain wall stands out. A Gaussian blur (kernel size from Settings) is applied first.

- **Binarize toggle button** (shortcut: `Ctrl+B`): Switches the image tabs between raw and binarized views.
- **`Threshold type` dropdown:** Choose between:
  - **Custom** – use the manually set threshold value (0–255).
  - **Otsu** – compute the optimal threshold automatically with Otsu's method.
- **`Threshold Value` spin box:** The custom threshold (default: 144). Disabled while Otsu is selected. Changing it re-binarizes immediately when the toggle is on.
- **`Inverse` checkbox:** Inverts the binary image — use it when the domain/background polarity is opposite to the default.
- **`Current Threshold Value` label:** Shows the threshold actually used for the active tab (informative for Otsu, where the value is computed per image).
- **`Plot Histogram` button:** Opens a Matplotlib window with the pixel-intensity histogram of the (blurred) current image and a vertical line marking the threshold.

### Direction Selection

**Group box labeled "Select Direction"**

Used by `BUBBLE_DIRECTIONAL` and `ARBITRARY_STRUCTURE`. Defines the line along which domain wall displacement is measured.

- **`Load` button:** Copies the endpoints you clicked on the active image tab into the coordinate spin boxes.
- **`Point 1` / `Point 2` spin boxes:** The `(x, y)` pixel coordinates of the line endpoints. You can also type values directly.
- **`Width` dial:** Sets the **outline** — extra parallel lines generated around the main line (offsets of ±2, ±4, … pixels). Displacement is averaged over all of them for better statistics. Range 1–199 in steps of 2; at 1 only the main line is used, and the green parallel lines are drawn on the image when the value exceeds 2.
- **`Set` button:** Draws/applies the line (red) and its parallels (green) on the displayed images.
- **`Set to all` checkbox:** If checked, the line is applied to **all** image tabs; otherwise only to the current tab.
- **`Current Coordinates` label:** Live mouse position in pixels as you move over an image.

### Circular Expansion

**Group box labeled "Circular Expansion"**

Used exclusively by the `BUBBLE_CIRCLE_FIT` method.

- **`roundness` spin box** (default: 1.2): Filters detected contours by roundness, defined as perimeter² / (4π × area). A perfect circle has roundness 1.0; increase the value to also accept more elliptical domains.
- **`Select Domain` text field:** Optional filter — only track domains near the given centers, formatted as `(x1,y1);(x2,y2);...`. Leave empty to track all detected domains.
- **`Auto select` button:** Detects domains in the loaded images and fills the field with their centers automatically.
- **`Show Fit` button:** Opens a new tab showing all detected domain contours with their fitted enclosing circles drawn in red, so you can verify detection before calculating.

### Domain Center Selection

**Group box labeled "D Center Select"**

- **`center_select` text field** (default: `255,255`): The approximate pixel center of the bubble domain, as `x,y`. The `BUBBLE_DIRECTIONAL` method uses it to pick the correct closed contour out of everything in the binarized image.

### Calculation & Export

- **`Calculate Current` button:** Computes displacement for the currently loaded image set and prints mean displacement and standard deviation (in scaled physical units) to the text output area. The file names do not need to code the pulse width — load any image sequence, confirm the order if asked, and measure each amplitude manually one folder at a time.
- **`Calculate All` button:** Batch-processes every amplitude folder sitting next to the loaded folder, using a pool of 4 worker processes. While running, the button reads "Stop" — click it again to terminate the workers. Requires coded pulse widths in every image name; folders without them are checked first and listed in a warning instead of starting the run.
- **`Bulk Calculate` button:** Goes one directory level higher and processes every experiment folder whose name ends in `mT`, each in turn like `Calculate All` (currently single-purpose for field datasets; disabled unless enabled in code).
- **`Result Save Folder` field** (default: `DisplacementTime`): Name of the subfolder (created next to the processed folders) where result files are written.
- **`Overwrite if save folder exists` checkbox:** If unchecked and the save folder already exists, you get a warning and the run is skipped instead of overwriting.
- **`Plot Current` button:** Reads the result files from the save folder and plots displacement-vs-pulse-width curves (one series per amplitude) with Matplotlib.
- **`Get edge` button:** Builds the domain-wall edge image from the loaded set and opens it in a new tab labelled `edges`.
- **Text output area:** Shows calculation results and status messages.

### Menu Bar

| Menu | Item | Action |
|---|---|---|
| **File** | Select Folder (`Ctrl+F`) | Same as the `Select Folder` button |
| **View** | Image Viewer (`Ctrl+D`) | Opens the image browser window |
| **View** | Reorder Images | Re-opens the manual ordering window for the loaded images |
| **View** | Settings | Opens the settings window |
| **Help** | Documentation | Opens the project GitHub page in your browser |
| **Help** | About | Shows version, author, and citation info |

---

## Measurement Workflows

### Bubble Directional Workflow

For circular bubble domains expanding around a center.

1. **Load images** of the bubble domain sequence.
2. **Select `BUBBLE_DIRECTIONAL`** in the measurement type dropdown.
3. **Set the domain center** in the `D Center Select` field (approximate bubble center in pixels).
4. **Configure binarization:** toggle the binarized view on, choose Otsu or Custom threshold (flip `Inverse` if needed) until the wall is cleanly separated.
5. **Draw a direction line** crossing the wall:
   - Left-click twice on an image tab to place the endpoints.
   - Click `Load` to copy them into the spin boxes.
   - Set the `Width` dial for parallel-line averaging.
   - Click `Set` (tick `Set to all` first if you want it on every tab).
6. **Calculate:** `Calculate Current` for a single check, or `Calculate All` for the full batch.
7. **Inspect:** `Plot Current` to visualize the displacement curves.

### Arbitrary Structure Workflow

For non-circular, irregular domain patterns.

1. **Load images** and select `ARBITRARY_STRUCTURE`.
2. **Configure binarization** as above.
3. **Draw a direction line** across the region where the wall moves (same procedure as step 5 above). This method finds the wall with Canny edge detection and traces its contour, then intersects it with your line.
4. **Calculate** and inspect results.

### Bubble Circle Fit Workflow

For circular bubble domains (e.g. nucleation-and-expansion experiments).

1. **Load images** and select `BUBBLE_CIRCLE_FIT` (no direction line or center needed).
2. **Configure binarization.**
3. **Set `roundness`** (default 1.2) to match your domains.
4. **Optionally filter domains:** click `Auto select`, or type centers manually into `Select Domain`.
5. **Click `Show Fit`** to verify circles are fitted to the right domains.
6. **Calculate.** Displacement here is the change in fitted circle radius between frames.

### Manual Workflow (no coded pulse widths needed)

For measurement systems that do not hardcode the pulse width into image names. Measure each amplitude folder by hand, one at a time:

1. **Load the folder** — any image names work (e.g. `frame1.png ... frame10.png`). If the order cannot be determined from the names, the Confirm image order window appears: check the predicted order with the preview, rearrange if needed, press OK.
2. **Set measurement parameters** as in the workflows above.
3. **Click `Calculate Current`** and note the Displacement and Standard Deviation from the results box.
4. **Repeat** for each amplitude folder. `Calculate All` and `Plot Current` are unavailable in this mode because they need coded pulse widths to build the displacement-vs-pulse-width tables.

---

## Settings Window

Accessed via `View > Settings`. Values take effect only after clicking **Set** — the button stays disabled until you change something.

| Setting | Description | Default (`settings/default.set`) |
|---|---|---|
| **Scale** | Physical distance per pixel (e.g. µm/pixel). Pixel displacements are multiplied by this. Calibrate it for your microscope. | 0.67114093 |
| **Gaussian Blur Kernel** | Kernel size of the Gaussian blur applied before binarization. Must be odd. | 25 |
| **Max Width** | Crop images to this width in pixels (0 = no cropping). | 0 |
| **Max Height** | Crop images to this height in pixels (0 = no cropping). | 512 |
| **Image format** | Only files with this extension are loaded: PNG, JPEG/JPG, or WebP. | png |

- **Load Preset:** Opens a file dialog (starting in `settings/`) to load a previously saved `.set` file into the dialog.
- **Save Preset:** Opens a file dialog to save the current dialog values as a `.set` file.
- **Set:** Applies the dialog values to the running application.

---

## Image Viewer

A browser window (`Ctrl+D` or `View > Image Viewer`) for scanning through a dataset folder by folder. It opens pre-synced with the main window's folder and follows the main window's measurement settings live.

- **Select Folder:** Pick any measurement folder — all its sibling amplitude folders are collected automatically for browsing.
- **Previous / Next:** Step to the adjacent folder.
- **Skip Backward / Skip Forward:** Jump to the same run number at the next lower/higher amplitude.
- **Process Images checkbox:** When ticked, the viewer computes and shows the domain-wall edge image using the main window's current measurement type and binarization settings (computed locally, not via the API). When unticked, it shows a raw image (the third image of the folder).
- **Directory label:** Shows the folder currently displayed.

This is useful for quickly flipping through experimental folders to find the right one before loading it into the main window.

---

## REST API

MODVORTEx embeds a Flask app served by waitress on `http://localhost:5454`. Click **`Start API`** to launch it (the button reads "Stop API" while running); analysis runs in a process pool sized to your physical CPU count.

> **Important:** every endpoint uses the measurement type and binarization settings currently configured in the GUI. Set those first.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/avg_dis` | Send image data, get the mean displacement |
| `POST` | `/api/dis` | Send image data, get the full displacement list |
| `POST` | `/api/edge` | Send image data, get the edge image |
| `POST` | `/api/edge_from_path` | Send a folder path, get the edge image computed from that folder |
| `GET` | `/api/current_state` | Get the current measurement type and binarization settings (jsonpickle-encoded) |
| `GET` | `/api/current_avg_dis` | Mean displacement of the images currently loaded in the GUI |
| `GET` | `/api/current_dis` | Full displacement list of the images currently loaded in the GUI |
| `GET` | `/api/current_edge` | Edge image of the images currently loaded in the GUI |
| `GET` | `/` | Health check (returns the request environment keys) |

Image payloads are JSON objects with keys `image0`, `image1`, … mapping to nested lists of uint8 grayscale pixels.

### Example: Python

```python
import requests
import numpy as np

# Grayscale uint8 images, e.g. loaded with cv2.imread(path, cv2.IMREAD_GRAYSCALE)
images = [np.zeros((512, 512), dtype=np.uint8) for _ in range(5)]
data = {f'image{i}': img.tolist() for i, img in enumerate(images)}

# Mean displacement with the GUI's current settings
response = requests.post('http://localhost:5454/api/avg_dis', json=data)
print("Average Displacement:", response.json())

# Edge image with the GUI's current settings
response = requests.post('http://localhost:5454/api/edge', json=data)
edge_image = np.array(response.json(), dtype=np.uint8)
```

---

## File Format & Data Export

### Image file naming

The pulse width is parsed out of each image file name: the number before the unit letters is the pulse width, and `p` acts as the decimal point. Digits after the unit (with or without an underscore) are the repeat number, not part of the pulse width. Examples:

- `1p0us.png` → 1.0 (µs), repeat 0
- `500ns.png` → 500 (ns), repeat 0
- `0p44s_001.png` → 0.44 (s), repeat 1
- `0p44s10.png` → 0.44 (s), repeat 10

All repeats in one folder must share the same pulse width (e.g. `0p44s.png, 0p44s_001.png ... 0p44s_005.png`). Any of these layouts work — the photos are always sorted with the lowest number first, which becomes the first tab and the reference image:

- `0p44s.png, 0p44s_001.png ... 0p44s_005.png`
- `0p44s_000.png ... 0p44s_005.png`
- `0p44s_001.png ... 0p44s_006.png` (no bare file — `_001` is first)
- `0p44s.png, 0p44s_1.png ... 0p44s_5.png` (no zero-padding — `_2` still sorts before `_10`)
- `0p44s0.png ... 0p44s10.png` (no underscore)

A file with no repeat digits counts as repeat 0, so `0p44s.png` always comes before `0p44s_001.png`.

> **Warning - no unit conversion:** The program only reads the number and ignores the unit letters. It does not convert ns, us, ms, s into each other.
> Keep the same unit for all photos in your dataset, especially when using Calculate All batch processing. If you mix `500ns.png` (saved as 500) with `1p0us.png` (saved as 1.0), they will be plotted together as 500 and 1.0 and your velocity will be wrong.

> **Warning - don't mix bare and `_000`:** Never put `0p44s.png` and `0p44s_000.png` in the same folder — both count as repeat 0. Pick one naming style per folder.

Names without any coded pulse width (e.g. `frame1.png`) load fine for manual `Calculate Current` measurements — the Confirm image order window will ask you to verify the order. They cannot be used with `Calculate All`: pressing it first scans all folders and warns about any folder whose names lack a coded pulse width, before any calculation starts.

### Output files

One result file is written per amplitude, named `{voltage}{unit}.txt` (e.g. `25V.txt`, `30mT.txt`), inside the save folder. The first line is a comment recording the measurement parameters:

```
# point2 : (x,y),  point1 : (x,y),  DCenter : (x,y),  Binarize : <value>,  outline : <n>
```

The remaining lines are tab-separated with three columns:

| pulse_width | displacement | std |
|---|---|---|

```
1.0	0.45	0.03
2.0	0.92	0.05
3.0	1.38	0.04
```

`displacement` is the mean wall displacement in scaled physical units (pixels × Scale); `std` is the standard deviation over the parallel lines and repeated runs at the same pulse width.

### Velocity calculation

Import the displacement-vs-pulse-width tables into your analysis tool of choice (Origin, Python, MATLAB, …) and extract the velocity from the displacement curve.

---

## Troubleshooting

| Problem | Possible Cause | Solution |
|---|---|---|
| Tabs stay empty after `Load` | Path box doesn't point at the image folder, or no files match the configured image format | Use `Select Folder` on a file inside the image folder; check `View > Settings > Image format` |
| Binarized image is all black or all white | Wrong threshold or polarity | Try Otsu instead of Custom (or vice versa), adjust the threshold value, or tick `Inverse` |
| `Calculate Current` prints `nan` | No wall found where expected | Check binarization quality; make sure the direction line actually crosses the wall; for circle fit, relax `roundness` |
| `Get edge` shows an empty image | No detectable wall in the loaded set | Same checks as above; an all-blank edge image simply means nothing was detected |
| API won't start | Port 5454 already in use | Stop the other process using port 5454, or change the port in `api.py` |
| Line drawing does nothing | Clicks land outside the image label | Click on the image itself, not the tab bar or panel controls |

---

## Citing MODVORTEx

If MODVORTEx is useful in your research, please cite:

1. Rakhul Raj et al., *Meas. Sci. Technol.* **36** (2025) 015001 [DOI: 10.1088/1361-6501/ad8beb](https://dx.doi.org/10.1088/1361-6501/ad8beb)
2. Rakhul Raj et al., *Meas. Sci. Technol.* **34** (2023) 085001 [DOI: 10.1088/1361-6501/acd39d](https://dx.doi.org/10.1088/1361-6501/acd39d)

---

> **GitHub:** [https://github.com/RakhulR/MODVORTEx](https://github.com/RakhulR/MODVORTEx)
