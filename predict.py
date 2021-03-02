from recognition import Recognizer

recognizer = Recognizer('./recognition.json')
recognizer.load()
result = recognizer.predict('./predictions/')
print(result)