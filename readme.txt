Project Introduction: 

Data acquisition:
Installation:
1. Basic environment requirements for running the code: Minimum requirement: Python 3.6; Recommended version: Python 3.7+
2. Core dependent libraries:
pandas >= 1.3.0 (for reading/writing Excel/CSV files and data processing)
numpy >= 1.21.0 (for numerical calculations, Fourier transforms, etc., core operations)
openpyxl >= 3.0.0 (implicit dependency) (pandas needs to read xlsx files)
PyTorch >= 1.12.0 [neural network framework (needs to match torchvision version)]
torchvision >= 0.13.0 (support for image processing and pre-trained models)
scikit-learn >= 1.1.0 [evaluation metrics (accuracy_score, etc.)]
matplotlib >= 3.5.0 (for visualization during training process)
Pillow >= 9.0.0 (for image display functionality) and so on.
3. Version compatibility notes: PyTorch must match the CUDA version; to prevent dependency conflicts, it is recommended to use virtual environments (venv/conda). 
Specific operations: 
If you want to obtain the final result quickly, you can skip step 1 (data processing) and directly use the provided dataset "resultce.csv" file, and run "train.py" and "Model Invocation.py" 
Steps:
1. Data preprocessing: Firstly, run the preprocess.py file. Adjust the original data directory in line 285 and the saved processed data directory in line 222 according to the actual situation.
Firstly, it will read the Excel file, delete the first row of headers, and rename the first eight columns to X0 to X7. Then, when processing the CSV file, only keep the columns X1, X3, X5, X7, and X9, truncate the rows with data to the same length, and remove the columns with too many missing values. Next, perform Fourier analysis on each segment of the signal, calculate the amplitude, energy, and peak frequency features, and then mix in randomly sampled sub-sample data.
Finally, add an ID label to each sample, and combine all the features to save as a CSV file. The output result is approximately 200 rows of data, each row having over 200 features, including the details of Fourier analysis, original signal statistics, and category encoding.
Since the Excel file has 24 groups, the final output is a approximately 4801-row data.csv file.
【Notes】
If you encounter a PermissionError, check if the file is occupied by another program.
If the compute_fourier_features fails, ensure that the input signal length > 0.
Adjust the parameters in process_xlsx_with_pandas according to the requirements (such as sampling rate) 
2. Run the 'train.py' file:
Data Preparation
Place the 'best_model.pth' file and the 'train.py' file in the same directory. Change the '.csv' file name in line 336 of the 'train.py' code to the '.csv' file processed in step 1 (or directly use the given dataset 'new_result.csv') 
Adjust the following parameters: config = {
'batch_size': 32,         # Batch size
'val_split': 0.2,         # Validation set ratio
'hidden_dims': [256,128,64,32],  # Hidden layer dimensions
'lr': 0.01,               # Learning rate
'weight_decay': 1e-5,     # L2 regularization coefficient
'epochs': 100,            # Training epochs
'save_path': './best_model.pth'  # Model save path }

Run the program and train for 100 rounds (adjust as needed) 
Output results:
Training process curve graph: training_curve.png
Best model weights: best_model.pth
Validation set evaluation report: classification_report 
【Notes】
Adjust the batch_size according to the hardware (reduce it when the GPU memory is insufficient)
If CUDA runs out of memory, try reducing the batch_size or use --no-cuda
Adjust hidden_dims and learning_rate to optimize the model performance 

3. Run the Model Invocation.py file:
Prepare the data
Make sure the file used in line 158 of the code is new_result.csv
Place the trained model weight file best_model.pth and the Model Invocation.py file in the same directory
Modify the mechanical hand photo file directory in line 214 of the Model Invocation.py file according to the actual situation 
Modify configuration parameters
In the function load_trained_model(), adjust the following parameters:
MODEL_PATH = "./best_model.pth"  # Path for model weights
NUM_CLASSES = 24                # Number of categories 
Running prediction 
# Enter the directory where the code is located in the command line cd /path/to/Model Invocation

# Execute the prediction script Model Invocation.py

Output result
The prediction results are saved as "prediction_results.csv"
Automatically display the corresponding image for the predicted category (make sure the image path is correct) 
【Notes】
If the image cannot be opened, check whether the file path contains Chinese characters or special characters.
Adjust the preprocessing parameters in predict_transform (such as standardization mean/standard deviation)
