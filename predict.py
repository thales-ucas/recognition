from recognition import Recognizer

recognizer = Recognizer('./recognition.json')
recognizer.load()
result = recognizer.predict('./predictions/')
print(result)

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, len(result), figsize=(9, 5))
for idx, obj in enumerate(result):
  img = plt.imread(obj['img'])
  axes[idx].imshow(img)
  axes[idx].set_title(obj['label'])
plt.show()