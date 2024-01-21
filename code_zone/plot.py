import seaborn as sns
import pickle
import matplotlib.pyplot as plt

path = 'saved_model/20200513-175920'

with open(f'{path}/history.pickle', 'rb') as f:
    history = pickle.load(f)
    history =


plt.plot(history['loss'])
plt.plot(history['val_loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
