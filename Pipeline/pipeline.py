import generate_balloons
import generate_personagens
from pathlib import Path
import cv2 as cv
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

image = Path('./generate_input/0019_jpg.rf.0b3de368698d2f066257fcf8dbe5a0a3.jpg')
generate_balloons.generateUnity(image)
generate_personagens.generateUnity(image)

balloons_path = Path('./generate_output_balloons_unity/speech_balloons')


balloons_list = list(balloons_path.iterdir())
def display_ballons(balloons_path):
    for balloon in balloons_path.iterdir():
    # display the ballons for debugging
        img = cv.imread(str(balloon))
        cv.imshow('image', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

# display_ballons(balloons_path)





