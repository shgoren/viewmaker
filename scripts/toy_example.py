visible_lines = 20
from torch import nn, autograd

import torch
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import defaultdict
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

from IPython import display
from IPython.display import HTML
from IPython.display import clear_output

from viewmaker.src.objectives.neutralad import NeuTraLADLoss


class Toy(nn.Module):

    def __init__(self, d=2, num_views=2, budget=0.2):
        super().__init__()
        self.d = d
        self.h = d * 3
        self.v = num_views
        self.budget = budget
        self.enc = nn.Sequential(nn.Identity(),
                                 nn.Linear(self.d, self.h, bias=False), nn.ReLU(),
                                 #                                  nn.Linear(self.h, self.h), nn.ReLU(),
                                 # #                                  nn.Linear(self.h, self.h), nn.ReLU(),
                                 nn.Linear(self.h, self.d, bias=False)
                                 )
        # TODO: uncomment this and add noise

        self.vms = nn.ModuleList([nn.Sequential(nn.Linear(self.d + 1, self.h), nn.ReLU(),
                                                nn.Linear(self.h, self.h), nn.ReLU(),
                                                nn.Linear(self.h, self.h), nn.ReLU(),
                                                nn.Linear(self.h, self.d)) for _ in range(self.v)])

        # self.vm = nn.Sequential(nn.Linear(self.d, self.h * self.v, bias=False), nn.ReLU(),
        #                         nn.Linear(self.h * self.v, self.d * self.v, bias=False))

    def add_noise_coordinate(self, x, num=1, bound_multiplier=1):
        # bound_multiplier is a scalar or a 1D tensor of length batch_size
        batch_size = x.size(0)
        shp = (batch_size, num)
        bound_multiplier = torch.tensor(bound_multiplier, device=x.device)
        noise = torch.rand(shp, device=x.device) * bound_multiplier
        return torch.cat((x, noise), dim=1)

    def forward(self, x, budget=None):
        x_n = self.add_noise_coordinate(x)
        view_residuals = torch.stack([vm(x_n) for vm in self.vms]).permute(1, 0, 2)  # b,v,d
        # view_residuals = self.vm(x).reshape(-1, self.v, self.d)
        view_residuals = self.get_delta(view_residuals, budget)
        x_tags = F.normalize(x + view_residuals.permute(1, 0, 2), dim=-1)
        x_tags_enc = F.normalize(self.enc(x_tags.reshape(-1, self.d)).reshape(self.v, -1, self.d), dim=-1)
        x_enc = F.normalize(self.enc(x), dim=-1)
        return x_enc, x_tags_enc, x_tags

    def get_delta(self, y_pixels, budget, eps=1e-4):
        '''Constrains the input perturbation by projecting it onto an L1 sphere'''
        delta = torch.tanh(y_pixels)  # Project to [-1, 1]
        if budget is not None:
            avg_magnitude = delta.abs().mean([1, 2], keepdim=True)
            max_magnitude = budget
            delta = delta * max_magnitude / (avg_magnitude + eps)
        return delta


def plot_2d_space_transformed(toy, ax, res=720):
    cmap = matplotlib.cm.get_cmap('twilight')
    rad2ratio = lambda r: (r + np.pi) / (2 * np.pi)
    for d in np.linspace(-np.pi, np.pi, res):
        x = np.sin(d)
        y = np.cos(d)
        trans, _, _ = toy(torch.tensor([[x, y]], dtype=torch.float32))
        c = rad2ratio(np.arctan2(*trans[0].detach().numpy()))
        ax.plot((0, x), (0, y), color=cmap(c))
        ax.plot((x, x * 1.2), (y, y * 1.2), color=cmap(rad2ratio(d)))


def prepare_animation(history, lines, encodings=True):
    def animate(step):
        if encodings:
            X_tag = history['X_tag_enc'][step]
            x = history['x_enc'][step]
        else:
            x = history['x'][0]
            X_tag = history['X_tag'][step]
        changed_lines = []
        for b_ in range(min(x.size(0), visible_lines)):
            for i_, v in enumerate(x[[b_]].tolist() + X_tag[:, b_].tolist()):
                line = lines[b_][i_]
                line.set_xdata([0, v[0]])
                line.set_ydata([0, v[1]])
                changed_lines.append(line)
        return changed_lines

    return animate

def make_animation(history, b, encodings, steps):
    fig = plt.figure(figsize=(6, 12))
    ax0 = fig.add_subplot(211)
    ax0.set_xlim(-1, 1)
    ax0.set_ylim(-1, 1)
    ax1 = fig.add_subplot(212)

    lines = []
    X_tag = history[0]
    x_enc = history['x_enc'][0]
    X_tag_enc = history['X_tag_enc'][0]
    for i in range(min(b, visible_lines)):
        sample_lines = []
        for i, v in enumerate(x_enc[[i]].tolist() + X_tag_enc[:, i].tolist()):
            lbl = "target" if i == 0 else i - 1
            if i==0:
                c = "r"
                a = 1
            else:
                c = 'b'
                a = 0.2
            sample_lines.append(
                ax0.plot([0, v[0]], [0, v[1]], label=lbl, color=c, alpha=a)[0])
        lines.append(sample_lines)
    ax1.plot(history['losses'], color='b')
    animate = prepare_animation(history, lines, encodings=encodings)
    ani = animation.FuncAnimation(fig,
                                  animate,
                                  frames=steps,
                                  interval=150,
                                  blit=True)

    display.display(HTML(ani.to_html5_video()))

def evaluation_plot(hist, ax, centers, n_views):
    cmap = {0: 'k', 1: 'b', 2: 'y', 3: 'g', 4: 'r', 5: 'purple'}

    x_enc = hist['x_enc'][-1].detach().numpy()
    X_tag = hist['X_tag'][-1].detach().numpy().reshape(-1, 2)
    y = KMeans(centers).fit_predict(x_enc)
    x = hist['x'][-1].detach().numpy()
    ax.scatter(x[:, 0], x[:, 1], c=list(map(cmap.get, y)))
    ax.scatter(X_tag[:, 0], X_tag[:, 1], c=list(map(cmap.get, np.stack([[y] * n_views]).ravel())), alpha=0.1)
    for v, l in zip(x_enc, y):
        ax.plot([0, v[0]], [0, v[1]], c=cmap[l])


def get_scheduler(num_steps):
    schedule = np.linspace(0.05, 1, num_steps)

    def scheduler(step):
        if step >= len(schedule):
            return None
        else:
            return schedule[step]

    return scheduler


def run(t, budget, b, lr, n_views=2, d=2, plot=False, steps=100, encodings=True, Loss=NeuTraLADLoss, centers=4,
        animate=False):
    toy = Toy(budget=budget, num_views=n_views)
    vm_optim = Adam(toy.vms.parameters(), lr=lr)
    enc_optim = Adam(toy.enc.parameters(), lr=lr)
    sch = get_scheduler(150)
    # c = ['r','b','y','k']
    losses = []
    history = defaultdict(list)
    torch.random.manual_seed(11)
    #     x = F.normalize(torch.rand(b, d))
    x, y = make_blobs(b + 100, centers=centers, random_state=10)
    x, x_test = x[:b], x[b:]
    y, y_test = y[:b], y[b:]
    x = F.normalize(torch.from_numpy(x).float())
    loss_parts = []
    history['x'].append(x)
    history['y'].append(y)
    history['x_test'].append(x)
    history['y_test'].append(y)
    for step in range(steps):
        with autograd.detect_anomaly():
            x_enc, X_tag_enc, X_tag = toy(x, sch(step))
            nad = Loss(x_enc, X_tag_enc, t=t)
            crit, parts = nad.get_loss(debug=True)
            #         if torch.isnan(x_enc[0][0]):
            #             print(x_enc, x, sch)
            losses.append(crit)
            history["X_tag"].append(X_tag)
            history["X_tag_enc"].append(X_tag_enc)
            history["x_enc"].append(x_enc)
            history["loss_parts"].append(parts)
            history["losses"].append(crit)
            vm_optim.zero_grad()
            enc_optim.zero_grad()
            crit.backward()
            vm_optim.step()
            if step % 2 == 0:
                enc_optim.step()

        if plot and step % plot == 0:
            clear_output(wait=True)
            fig = plt.figure(figsize=(15, 16))
            fig.suptitle(f"step - {step}")

            ax1 = fig.add_subplot(3, 2, 1)
            ax2 = fig.add_subplot(3, 2, 2)
            ax3 = fig.add_subplot(3, 2, 3)
            ax4 = fig.add_subplot(3, 2, 4)
            ax5 = fig.add_subplot(3, 2, 5)
            ax6 = fig.add_subplot(3, 2, 6)

            list_of_lists = [p.grad.detach().flatten().tolist() for p in toy.enc.parameters()]
            list_of_grads = [item for sublist in list_of_lists for item in sublist]
            ax1.hist(list_of_grads,
                     # bins=np.linspace(-1, 1, 100)
                     )
            ax1.set_title(f"enc")

            list_of_lists = [p.grad.detach().flatten().tolist() for p in toy.vms.parameters()]
            list_of_grads = [item for sublist in list_of_lists for item in sublist]
            ax2.hist(list_of_grads,
                     # bins=np.linspace(-1, 1, 100)
                     )
            ax2.set_title("vm")

            for i in range(min(b, visible_lines)):
                for i, v in enumerate(x_enc[[i]].tolist() + X_tag_enc[:, i].tolist()):
                    lbl = "target" if i == 0 else i - 1
                    if i == 0:
                        c, a = "r", 1
                    else:
                        c, a = 'b', 0.2
                    ax3.plot([0, v[0]], [0, v[1]], label=lbl, color=c, alpha=a)[0]
            ax3.set_xticks(np.linspace(-1, 1, 11))
            ax3.set_yticks(np.linspace(-1, 1, 11))
            ax4.plot(history['losses'], color='b')
            evaluation_plot(history, ax5, centers, n_views)
            plot_2d_space_transformed(toy, ax6)

            # idsiplay(fig)
            plt.show()
            plt.close(fig)
    if animate:
        make_animation(history, b, encodings, steps)

    return crit, history

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    seed_everything(10)
    loss, hist = run(t=1,
                     budget=1,
                     b=50,
                     lr=10 ** -1.421,
                     n_views=4,
                     d=2,
                     steps=301,
                     plot=110,
                     encodings=True)
    print(loss)
