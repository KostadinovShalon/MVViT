from matplotlib.widgets import Slider

from MVViT.models.transformers.mv_positional_encoding import MVSinePositionalEncoding
from mmdet.models.utils import SinePositionalEncoding
from matplotlib import pyplot as plt
import torch

dims = 256

pe = MVSinePositionalEncoding(dims, normalize=True)
sv_pe = SinePositionalEncoding(dims // 2, normalize=True)
views, h, w = 2, 25, 25
sv_masks = torch.zeros((1, h, w))
mv_sv_masks = torch.zeros((1, h * views, w))
masks = torch.zeros((1, views, h, w))
sv_pos_emb = sv_pe(sv_masks).squeeze(0).numpy()
mv_sv_pos_emb = sv_pe(mv_sv_masks).squeeze(0).numpy()
pos_emb = pe(masks).squeeze(0).numpy()

init_dim = 0

fig, axes = plt.subplots(2, 5)

gs0 = axes[0, 3].get_gridspec()
gs1 = axes[0, 4].get_gridspec()
for ax in axes[:, 3]:
    ax.remove()
for ax in axes[:, 4]:
    ax.remove()

dims_plotsv2i = axes[0, 0].imshow(sv_pos_emb[2 * int(init_dim)], cmap='jet', vmin=-1, vmax=1)
dims_plotsv2ip1 = axes[1, 0].imshow(sv_pos_emb[2 * int(init_dim) + 1], cmap='jet', vmin=-1, vmax=1)
dims_plotv0d2i = axes[0, 1].imshow(pos_emb[0][2 * int(init_dim)], cmap='jet', vmin=-1, vmax=1)
dims_plotv0d2ip1 = axes[1, 1].imshow(pos_emb[0][2 * int(init_dim) + 1], cmap='jet', vmin=-1, vmax=1)
dims_plotv1d2i = axes[0, 2].imshow(pos_emb[1][2 * int(init_dim)], cmap='jet', vmin=-1, vmax=1)
dims_plotv1d2ip1 = axes[1, 2].imshow(pos_emb[1][2 * int(init_dim) + 1], cmap='jet', vmin=-1, vmax=1)

ax_mv2_2i = fig.add_subplot(gs0[:, 3])
ax_mv2_2ip1 = fig.add_subplot(gs1[:, 4])

dims_plotmvd2i = ax_mv2_2i.imshow(mv_sv_pos_emb[2 * int(init_dim)], cmap='jet', vmin=-1, vmax=1)
dims_plotmvd2ip1 = ax_mv2_2ip1.imshow(mv_sv_pos_emb[2 * int(init_dim) + 1], cmap='jet', vmin=-1, vmax=1)


axcolor = 'lightgoldenrodyellow'
axes[0, 0].set_title("SV, dim 2i")
axes[1, 0].set_title("SV, dim 2i")
axes[0, 1].set_title("MV0, dim 2i")
axes[0, 2].set_title("MV1, dim 2i")
axes[1, 1].set_title("MV0, dim 2i+1")
axes[1, 2].set_title("MV1, dim 2i+1")
ax_mv2_2i.set_title("MVcat, dim 2i")
ax_mv2_2ip1.set_title("MVcat, dim 2i+1")
for ax in axes[:2, :3].flat:
    ax.margins(x=0)
    ax.label_outer()

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25)

# Make a vertically oriented slider to control the amplitude
axamp = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
dim_slider = Slider(
    ax=axamp,
    label="Dim 2i",
    valmin=0,
    valmax=dims // 2 - 1,
    valinit=init_dim,
    orientation="vertical",
    valstep=1
)


# The function to be called anytime a slider's value changes
def update(val):
    dims_plotsv2i.set_data(sv_pos_emb[2 * int(val)])
    dims_plotsv2ip1.set_data(sv_pos_emb[2 * int(val) + 1])
    dims_plotv0d2i.set_data(pos_emb[0][2 * int(val)])
    dims_plotv0d2ip1.set_data(pos_emb[0][2 * int(val) + 1])
    dims_plotv1d2i.set_data(pos_emb[1][2 * int(val)])
    dims_plotv1d2ip1.set_data(pos_emb[1][2 * int(val) + 1])
    dims_plotmvd2i.set_data(mv_sv_pos_emb[2 * int(val)])
    dims_plotmvd2ip1.set_data(mv_sv_pos_emb[2 * int(val) + 1])
    fig.canvas.draw_idle()


# register the update function with each slider
dim_slider.on_changed(update)
plt.show()
