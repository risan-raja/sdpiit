from FSketch import FSketch
import numpy as np


if __name__ == "__main__":
    dsaa = np.random.randint(879, size=(54655, 452))
    fs = FSketch(dsaa, 500, 319, 50, random_state=42)
    we = fs.create_sketch()
