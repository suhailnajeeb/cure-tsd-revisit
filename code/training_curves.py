import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "C:\\Data\\CURE-TSD\\models\\haze\\dehaze_trainingLog.csv"

df = pd.read_csv(csv_path)

#plt.figure()
df.plot(x = 'epoch', y = ['loss', 'val_loss'])
plt.title('Training Curve for Haze')
plt.show()

