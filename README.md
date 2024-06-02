# Comics Lib

Library created for the final project of the Computer Vision class of Insper

### Members
- Eric Possato
- Pedro Bittar Barão
- Rodrigo Patelli

### Usage
- Create a virtual environment and import the dependencies in the requirements.txt file
- Tesseract-ocr cannot be installed directly through pip, and should be downloaded from https://github.com/UB-Mannheim/tesseract/wiki on windows or following https://tesseract-ocr.github.io/tessdoc/Installation.html on other platforms. The script is setup to find the default windows installation path, otherwise must be changed at pipeline/text_extractor.py
- Inside the Pipeline/comics_lib.py file to have access to the functions
- Each function in the library is explained in its respective file, signature and usage examples can be found in Pipeline/example.ipynb.

### Details
- The file Pipeline/example.ipynb contains usage of each function in the library
- Each function also saves the results in the Pipeline/generate_output_{} folders respective to the operation performed

### Observations

- The folders Pipeline/training_output_{}, contain the trained models used for detection of balloons and characters. These models were trained inside the BalloonsDetectionTraining and CharactersDetectionTraining folders, respectively.

- The folder Pipeline/generate_input already contains some images for testing the functions. The images and are used for educational purposes only.