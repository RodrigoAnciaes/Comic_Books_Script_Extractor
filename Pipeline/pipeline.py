import generate_balloons
import generate_personagens
from pathlib import Path

image = Path('./generate_input/0010_jpg.rf.cc43fbe68c0feb90d0db546f23325db6.jpg')
generate_balloons.generateUnity(image)
generate_personagens.generateUnity(image)