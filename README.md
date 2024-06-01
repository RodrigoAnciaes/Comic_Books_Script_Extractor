# Comics Lib

Library created for the final project of the Computer Vision class of Insper

### Members
- Eric Possato
- Pedro Bittar Barão
- Rodrigo Patelli

### Usage
- Create a virtual environment and import the dependencies in the requirements.txt file
- Tesseract-ocr cannot be installed directly through pip, and should be downloaded from https://github.com/UB-Mannheim/tesseract/wiki on windows or following https://tesseract-ocr.github.io/tessdoc/Installation.html on other platforms. The script is setup to find the default windows installation path, otherwise must be changed at pipeline/text_extractor.py
- Import the comics_lib.py file to have access to the functions
- Each function in the library is explained in its respective file and usage examples can be found in example.ipynb.

### Details
- The file example.ipynb contains usage of each function in the library
- Each function also saves the results in the Pipeline/generate_output_{} folders respective to the operation performed
