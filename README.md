### Readme File

**Title: Code Explanation for Tunnel Face Crack Detection and GSI Prediction using U-Net and MLP**

**I. Project Overview**  
This project aims to detect cracks in tunnel face images using a U-Net network, extract crack features (number, length, width, density), and predict the Geological Strength Index (GSI) based on these features using an MLP model.


**II. Environment Configuration**  
1. **Python Version**: Python 3.8 or higher is recommended.  
2. **Required Libraries**:  
   - **PyTorch**: Deep learning framework for U-Net and MLP.  
     Installation: `pip install torch`  
   - **NumPy**: Numerical computation.  
     Installation: `pip install numpy`  
   - **Pandas**: Data processing and analysis.  
     Installation: `pip install pandas`  
   - **OpenCV-Python**: Image processing (reading, displaying, preprocessing).  
     Installation: `pip install opencv-python`  
   - **Scikit-learn**: Machine learning toolkit for data preprocessing and model evaluation.  
     Installation: `pip install scikit-learn`  
   - **PyQt5**: GUI library for UI design.  
     Installation: `pip install PyQt5`  


**III. Code Structure**  
- **DeepCrack_Image**: Contains tunnel face images and annotations for U-Net training/testing (format: .jpg).  
- **geological_data**: Stores crack geometric parameters (number, length, width, density) and GSI values for MLP training/testing (format: .xlsx).  
- **unet_rock_crack.pth**: Saved U-Net model.  
- **mlp_model.pth**: Saved MLP model.  
- **U-Net model.py**: Script for training the U-Net model.  
- **GSI-model.py**: Script for training the MLP model.  
- **Interface.ui**: Qt-based graphical interface design.  
- **GSI_Interface.py**: Main script to run the GUI application.  


**IV. Data Preparation**  
1. Organize collected tunnel face images into training and testing sets under the `DeepCrack_Image` folder. Ensure annotations are properly formatted for parsing during training.  
2. Place .xlsx files containing crack parameters (number, length, width, density) and GSI values in the `geological_data` folder.  


**V. Model Training**  
1. **U-Net Model Training**:  
   - Run: `python "U-Net model.py"`  
   - Training logs (loss, validation metrics) will be displayed.  
   - The trained model is saved as `unet_rock_crack.pth`.  

2. **MLP Model Training**:  
   - Ensure the U-Net model is trained and crack features are extracted.  
   - Run: `python "GSI-model.py"`  
   - Training metrics will be displayed.  
   - The trained model is saved as `mlp_model.pth`.  


**VI. UI Interface Operation**  
1. **Launch the Application**:  
   Ensure all dependencies are installed. Run:  
   ```bash  
   python GSI_Interface.py  
   ```  

2. **Load an Image**:  
   - Click `Open Image` (button linked to `openImage` method).  
   - Select a .jpg or .png tunnel face image.  
   - The image will be displayed at 270x270 pixels in the `label` area (converted to RGB for correct color display).  

3. **Crack Detection with U-Net**:  
   - Click `Open Image` (`open_2` button, linked to `select_image`).  
   - Load an image for crack detection (displayed in `label_18`).  
   - Click `Calculate` (`jisuan_2` button, linked to `detect_image`):  
     - The image is preprocessed and passed through the U-Net model.  
     - The binary crack mask is displayed in `label_19`.  
     - Crack features (number, length, width, occupancy) are calculated and filled into input fields (`lineEdit` to `lineEdit_4`).  

4. **Predict GSI**:  
   - Ensure crack features are filled in `lineEdit` (number), `lineEdit_2` (avg. length), `lineEdit_3` (avg. width), `lineEdit_4` (occupancy).  
   - Click `Predict` (linked to `predict` method).  
   - The MLP model predicts GSI based on the input features, and the result is shown in `Result_label`.  


**VII. Notes**  
1. Verify all data and model paths are correctly configured to avoid errors.  
2. Adjust model hyperparameters (e.g., U-Net layers, MLP neurons) or reduce batch sizes if facing memory constraints.  
3. This code serves as an example and may require adjustments for specific datasets or requirements.  

 

**Note**: Replace placeholders (e.g., `your_script_name.py`) with actual filenames. Ensure all paths and dependencies are correctly set before running the application.
