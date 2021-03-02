from recognition import Recognizer

recognizer = Recognizer('./recognition.json')
recognizer.train()
recognizer.save()

print('done')