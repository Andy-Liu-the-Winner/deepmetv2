from utils import load
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import mplhep as hep
plt.style.use(hep.style.CMS)

parser = argparse.ArgumentParser()
parser.add_argument('--ckpts', default='ckpts',
                    help="Name of the ckpts folder")

args = parser.parse_args()

loss = np.array([[2226.81, 1737.21],
[1783.94, 1683.03],
[1742.43, 1659.49],
[1717.74, 1651.75]])

train_loss = loss[:,0]
print(train_loss)
val_loss = loss[:,1]
print(val_loss)
x = np.arange(1,len(train_loss)+1)
plt.plot(x,train_loss,label='train')
plt.plot(x,val_loss,label='val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(args.ckpts+'loss.png')