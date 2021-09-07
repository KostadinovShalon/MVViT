from matplotlib.widgets import Slider

from MVViT.models.transformers.mv_positional_encoding import MVSinePositionalEncoding
from mmdet.models.utils import SinePositionalEncoding
from matplotlib import pyplot as plt
import torch

dims = 256

pe = MVSinePositionalEncoding(dims, normalize=True)
views, h, w = 2, 22, 22
masks = torch.zeros((1, views, h, w))
pos_emb = pe(masks).squeeze(0).numpy()

init_dim = 0

fig, (ax1, ax2) = plt.subplots(1, 2)
dims_plot = ax1.imshow(pos_emb[0][2*int(init_dim)], cmap='gray', vmin=0, vmax=1)
dims_plot2 = ax2.imshow(pos_emb[1][2*int(init_dim)], cmap='gray', vmin=0, vmax=1)
axcolor = 'lightgoldenrodyellow'
ax1.margins(x=0)
ax2.margins(x=0)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25)

# Make a vertically oriented slider to control the amplitude
axamp = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
dim_slider = Slider(
    ax=axamp,
    label="Dim 2i",
    valmin=0,
    valmax=dims//2-1,
    valinit=init_dim,
    orientation="vertical",
    valstep=1
)


# The function to be called anytime a slider's value changes
def update(val):
    dims_plot.set_data(pos_emb[0][2*int(val)])
    dims_plot2.set_data(pos_emb[1][2*int(val)])
    fig.canvas.draw_idle()


# register the update function with each slider
dim_slider.on_changed(update)
plt.show()
