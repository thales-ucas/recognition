from recognition import Recognizer

recognizer = Recognizer('./recognition.json')
recognizer.load()
recognizer.predict('./predictions/')