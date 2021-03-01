from recognition import Recognizer

recognizer = Recognizer()
recognizer.train('./images/')
recognizer.predict('./predict/1.png')