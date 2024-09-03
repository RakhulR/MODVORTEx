# MODVORTEx: A Tool for Automating Magnetic Domain Wall Velocity Analysis

**MODVORTEx** is a software tool designed for analyzing magnetic domain wall motion and calculating domain wall from a series of magnetic domain microscopy images. It provides different methods for tracking domain wall displacement and generating displacement time curve for calculating velocity data. For quasistatic domain wall velocity measurements, by providing measurement parameters (depending on the measurement type) MODVORTEx calculates the magnetic displacment-time curves for all magnetic pulse amplitudes. 

## Features

* **Multi-Method Displacement Calculation:**  MODVORTEx supports various approaches for determining domain wall displacement, including:
    * **Bubble Domain Circle Fit:**  Fits circles to bubble domains and tracks changes in radius.
    * **Bubble Domain Directional:** Tracks displacement along a user-defined line of a closed domain structure
    * **Arbitrary Structure:**  Traces the displacement of domain walls in complex, non-circular structures in the given direction.
* **Binarization Options:**  Allows users to binarize images using either Otsu's thresholding method or a custom threshold value.
* **Edge Detection:**  Generates edge images to visualize and analyze domain wall boundaries.
* **Data Export:** Exports displacement-time curves as text data files for further analysis.
* **Graphical User Interface (GUI):**  Provides a user-friendly interface for loading images, defining analysis parameters, and visualizing results.
* **Image Viewer:**  Includes a dedicated image viewer for browsing and inspecting microscopy image series.
* **REST API:** Offers a RESTful API for integrating MODVORTEx with external scripts or applications.

## Getting Started

1. **Installation:** 
   - Download and extract the zip file and execute MODVORTEx

3. **Using the GUI:**
   - Load your folder containing microscopy images (only PNG format for now).
   - Define measurement type (bubble domain, arbitrary structure).
   - Set binarization parameters.
   - Define measurement points (for directional measurements).
   - Calculate domain wall displacement and export data.
      - This can be automated for a set of images and field amplitudes 
	** For more details refer the documentation/user-guide (available soon)**
## API Usage

MODVORTEx's REST API enables automation and remote control and integration with external applications and measurement systems. 

**Example (using Python's `requests` library):**

```python
import requests
import numpy as np

# Assuming MODVORTEx's API is running on localhost:5454

images = [np.random.rand(512, 512) * 255 for _ in range(5)]  # Sample images (replace with your actual data)
data = {f'image{i}': img.tolist() for i, img in enumerate(images)}

# Get average displacement
response = requests.post('http://localhost:5454/api/avg_dis', json=data)
avg_displacement = response.json()
print("Average Displacement:", avg_displacement)

# Get edge image
response = requests.post('http://localhost:5454/api/edge', json=data)
edge_image_list = response.json()
edge_image = np.array(edge_image_list, dtype=np.uint8)
# ... process the edge_image ...
```

**See the code documentation/user-guide for details on the API endpoints and available methods.(available soon)** 

## Contributing

Contributions to MODVORTEx are welcome! 
- Report issues and bugs.
- Submit feature requests.
- Contribute code enhancements.

