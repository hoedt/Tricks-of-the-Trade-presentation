import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def centred(x: np.ndarray, sig_dims: int = 0, axis: tuple = None):
    """ 
    Get a centred version of a given dataset.

    Parameters
    ----------
    x : np.ndarray
        Array with at least 2 dimensions holding the dataset to be centred.
        First dimension is assumed to be the number of samples.
    sig_dims : int, optional
        Number of dimensions of the signal in the dataset.
        This allows to compute the mean over the entire signal,
        rather than over each entry in the signal individually.
    axis : tuple, optional
        Axes to compute the mean over.
        If `axis` is specified, `sig_dims` will be ignored.
    
    Returns
    -------
    x_c : np.ndarray
        Array like `x`, but centred to have zero mean.
    """
    x = np.atleast_2d(x)
    if axis is None:
        axis = (0, *range(-1, -1 - sig_dims, -1))
    
    return x - np.mean(x, axis, keepdims=True)


def normalised(x: np.ndarray, sig_dims: int = 0, eps: float = 1e-5):
    """
    Get a normalised version of a given dataset.
    """
    axis = (0, *range(-1, -1 - sig_dims, -1))
    x_c = centred(x, axis=axis)
    var = np.mean(x_c * x_c, axis, keepdims=True)
    return x_c / (var + eps) ** .5


def _generalised_eigh(arr: np.ndarray):
    size = np.prod(arr.shape[:arr.ndim // 2])
    l, u = np.linalg.eigh(arr.reshape(size, size))
    l = l.reshape(arr.shape[:arr.ndim // 2])
    u = u.reshape(arr.shape)
    return l, u

def _whitened_flat(x: np.ndarray, zca: bool = False, eps: float = 1e-5):
    _x_c = centred(x, sig_dims=0)
    x_c = _x_c.reshape(len(_x_c), -1)
    cov = x_c.T @ x_c / len(x_c)
    l, u = np.linalg.eigh(cov)
    x_rot = x_c @ u

    x_w = x_rot / np.sqrt(l + eps)
    if zca:
        x_w = x_w @ u.T
    
    return x_w.reshape(_x_c.shape)

def _whitened(x: np.ndarray, zca: bool = False, sig_dims: int = 0, eps: float = 1e-5):
    axis = (0, *range(-1, -1 - sig_dims, -1))
    _x_c = centred(x, axis=axis)
    
    axis_dual = range(1, _x_c.ndim - sig_dims)
    x_c = np.moveaxis(_x_c, axis_dual, range(-len(axis_dual), 0))
    x_c = x_c.reshape(np.prod(x_c.shape[:sig_dims + 1]), -1)

    cov = x_c.T @ x_c / len(x_c)
    l, u = np.linalg.eigh(cov)
    x_rot = x_c @ u

    x_w = x_rot / np.sqrt(l + eps)
    if zca:
        x_w = x_w @ u.T
    
    return x_w.reshape(_x_c.shape)


def whitened(x: np.ndarray, zca: bool = False, sig_dims: int = 0, eps: float = 1e-5):
    """
    Get a whitened version of a given dataset.
    """
    axis = (0, *range(-1, -1 - sig_dims, -1))
    x_c = centred(x, axis=axis)

    num_features = np.prod(x_c.shape[1:-sig_dims])
    num_samples = x_c.size // num_features
    cov = np.tensordot(x_c, x_c, axes=(axis, axis)) / num_samples
    l, u = _generalised_eigh(cov)

    axis_dual = range(1, x_c.ndim - sig_dims)
    x_rot = np.tensordot(x_c, u, axes=(axis_dual, range(len(axis_dual))))
    x_rot = np.moveaxis(x_rot, range(-len(axis_dual), 0), axis_dual)
    
    x_w = x_rot / (l.reshape(l.shape + (1, ) * sig_dims) + eps) ** .5
    if zca:
        x_w = np.tensordot(x_w, u, axes=(axis_dual, range(-len(axis_dual), 0)))
        x_w = np.moveaxis(x_w, range(-len(axis_dual), 0), axis_dual)
    
    return x_w


@torch.no_grad()
def evaluate(model, samples, metrics):
    model.eval()

    results = {k: [] for k in metrics}
    for x, y in samples:
        pred = model(x)
        for k, func in metrics.items():
            res = func(pred, y)
            results[k].append(res.item())
        
    return results


@torch.enable_grad()
def update(model, samples, loss_func, optimiser):
    model.train()

    errs = []
    for x, y in samples:
        pred = model(x)
        err = loss_func(pred, y)
        errs.append(err.item())

        optimiser.zero_grad()
        err.backward()
        optimiser.step()
    
    return errs



def plot_statistics(x: np.ndarray, bounds: tuple = None,
                    cmap: str = 'cividis', fig=None):
    if fig is None:
        fig = plt.figure()
    if bounds is None:
        print(np.quantile(x, [0., .01, .05, .95, .99, 1.]))
        bounds = np.quantile(x, [0., 1.])
    
    ax1, ax2 = fig.subplots(2, 1)
    min_val, max_val = bounds
    is_valid_rgb = (x.dtype == np.uint8) or (min_val >= 0. and max_val <= 1.)
    if is_valid_rgb:
        min_val = 0
        max_val = 255 if x.dtype == np.uint8 else 1.
    _x = np.moveaxis(x, 1, -1)  # channels first -> channels last

    if not is_valid_rgb and _x.shape[-1] == 3:
        new_x = np.ones((_x.shape[0], 2 * _x.shape[1], 2 * _x.shape[2]))
        new_x[:, :_x.shape[1], :_x.shape[2]] = _x[..., 0]
        new_x[:, _x.shape[1]:, :_x.shape[2]] = _x[..., 1]
        new_x[:, :_x.shape[1], _x.shape[2]:] = _x[..., 2]
        _x = new_x

    ax1.set_title("mean")
    im = ax1.imshow(_x.mean(0), vmin=min_val, vmax=max_val, cmap=cmap)
    ax1.axis('off')
    if not is_valid_rgb:
        axin = inset_axes(ax1, width="5%", height="100%",
                        loc="center right", borderpad=-1.5)
        plt.colorbar(im, cax=axin)

    ax2.set_title("std")
    im = ax2.imshow(_x.std(0), vmin=0, cmap='viridis')
    ax2.axis('off')
    axin = inset_axes(ax2, width="5%", height="100%",
                      loc="center right", borderpad=-1.5)
    plt.colorbar(im, cax=axin)

    return fig



def plot_samples(x: np.ndarray, nrows: int = 2, ncols: int = 5, cmap='cividis',
                 bounds: tuple = None, fig=None):
    """
    Create plot with samples and statistics of image dataset.

    Parameters
    ----------
    x : np.ndarray
        The dataset to visualise.
    nrows : int, optional
        The number of rows in the visualisation.
        Specifies (together with `ncols`) the number of samples to be shown.
    nrows : int, optional
        The number of columns in the visualisation.
        Specifies (together with `nrows`) the number of samples to be shown.
    cmap : str or cmap, optional
        The color map to use for visualising samples.
    low: float, optional
        Lower bound for clipping visualised pixel values.
    high: float, optional
        Upper bound for clipping visualised pixel values.
    fig : figure, optional
        The figure to draw the plot on.
    """
    if fig is None:
        fig = plt.figure()
    if bounds is None:
        bounds = np.quantile(x, [0., 1.])
    
    min_val, max_val = bounds
    is_valid_rgb = (x.dtype == np.uint8) or (min_val >= 0. and max_val <= 1.)
    if is_valid_rgb:
        min_val = 0
        max_val = 255 if x.dtype == np.uint8 else 1.
    _x = np.moveaxis(x, 1, -1)  # channels first -> channels last

    if not is_valid_rgb and _x.shape[-1] == 3:
        new_x = np.ones((_x.shape[0], 2 * _x.shape[1], 2 * _x.shape[2]))
        new_x[:, :_x.shape[1], :_x.shape[2]] = _x[..., 0]
        new_x[:, _x.shape[1]:, :_x.shape[2]] = _x[..., 1]
        new_x[:, :_x.shape[1], _x.shape[2]:] = _x[..., 2]
        _x = new_x
    
    axes = fig.subplots(nrows, ncols)
    for ax, xi in zip(axes.flat, _x[:nrows * ncols]):
        im = ax.imshow(xi, vmin=min_val, vmax=max_val, cmap=cmap)
        ax.axis('off')
    
    if not is_valid_rgb:
        cb_ax = fig.add_axes([.08, .2, .02, .6])
        fig.colorbar(im, cax=cb_ax)
        cb_ax.yaxis.set_ticks_position('left')

    return fig


def generate_plots(x_raw: np.ndarray, name: str):
    print("generating", name, "plots...")
    save_kwargs = {'bbox_inches': 'tight'}

    fig_raw = plot_samples(x_raw, cmap='gray')
    fig_raw.savefig('_'.join([name, "samples_raw.png"]), **save_kwargs)
    fig_stat1 = plot_statistics(x_raw, cmap='gray')
    fig_stat1.savefig('_'.join([name, "stats_raw.png"]), **save_kwargs)

    x_float = x_raw / 255.
    fig_float = plot_samples(x_float, cmap='gray')
    fig_float.savefig('_'.join([name, "samples_float.png"]), **save_kwargs)
    fig_stat2 = plot_statistics(x_float, cmap='gray')
    fig_stat2.savefig('_'.join([name, "stats_float.png"]), **save_kwargs)
    plt.close('all')

    # pixel normalisation

    x_mlp_c = centred(x_float)
    fig_centred = plot_samples(x_mlp_c)
    fig_centred.savefig('_'.join([name, "samples_pxl_centred.png"]), **save_kwargs)
    fig_centred = plot_statistics(x_mlp_c)
    fig_centred.savefig('_'.join([name, "stats_pxl_centred.png"]), **save_kwargs)

    x_mlp_n = normalised(x_float)
    fig_normed = plot_samples(x_mlp_n)
    fig_normed.savefig('_'.join([name, "samples_pxl_normalised.png"]), **save_kwargs)
    fig_normed = plot_statistics(x_mlp_n)
    fig_normed.savefig('_'.join([name, "stats_pxl_normalised.png"]), **save_kwargs)

    x_mlp_n = normalised(x_float)
    fig_normed = plot_samples(x_mlp_n, bounds=(-1.3, 3.0))
    fig_normed.savefig('_'.join([name, "samples_pxl_normalised_clipped.png"]), **save_kwargs)
    fig_normed = plot_statistics(x_mlp_n, bounds=(-1.3, 3.0))
    fig_normed.savefig('_'.join([name, "stats_pxl_normalised_clipped.png"]), **save_kwargs)

    x_mlp_pca = _whitened(x_float)
    fig_pca = plot_samples(x_mlp_pca, bounds=(-3, 3))
    fig_pca.savefig('_'.join([name, "samples_pxl_pca_clipped.png"]), **save_kwargs)
    fig_pca = plot_statistics(x_mlp_pca, bounds=(-3, 3))
    fig_pca.savefig('_'.join([name, "stats_pxl_pca_clipped.png"]), **save_kwargs)

    x_mlp_zca = _whitened(x_float, zca=True)
    fig_zca = plot_samples(x_mlp_zca, bounds=(-3, 3))
    fig_zca.savefig('_'.join([name, "samples_pxl_zca_clipped.png"]), **save_kwargs)
    fig_zca = plot_statistics(x_mlp_zca, bounds=(-3, 3))
    fig_zca.savefig('_'.join([name, "stats_pxl_zca_clipped.png"]), **save_kwargs)
    plt.close('all')

    # image normalisation

    x_cnn_c = centred(x_float, sig_dims=2)
    fig_centred = plot_samples(x_cnn_c)
    fig_centred.savefig('_'.join([name, "samples_img_centred.png"]), **save_kwargs)
    fig_centred = plot_statistics(x_cnn_c)
    fig_centred.savefig('_'.join([name, "stats_img_centred.png"]), **save_kwargs)

    x_cnn_n = normalised(x_float, sig_dims=2)
    fig_normed = plot_samples(x_cnn_n)
    fig_normed.savefig('_'.join([name, "samples_img_normalised.png"]), **save_kwargs)
    fig_normed = plot_statistics(x_cnn_n)
    fig_normed.savefig('_'.join([name, "stats_img_normalised.png"]), **save_kwargs)

    x_cnn_pca = _whitened(x_float, sig_dims=2)
    fig_pca = plot_samples(x_cnn_pca)
    fig_pca.savefig('_'.join([name, "samples_img_pca.png"]), **save_kwargs)
    fig_pca = plot_statistics(x_cnn_pca)
    fig_pca.savefig('_'.join([name, "stats_img_pca.png"]), **save_kwargs)

    x_cnn_zca = _whitened(x_float, zca=True, sig_dims=2)
    fig_zca = plot_samples(x_cnn_zca)
    fig_zca.savefig('_'.join([name, "samples_img_zca.png"]), **save_kwargs)
    fig_zca = plot_statistics(x_cnn_zca)
    fig_zca.savefig('_'.join([name, "stats_img_zca.png"]), **save_kwargs)
    plt.close('all')

    # data augmentation

    fig, axes = plt.subplots(2, 5)
    transformations = [
        None,
        transforms.CenterCrop(16),         # zoom in
        transforms.Pad(8),                 # zoom out
        transforms.Resize(14),             # pixelate
        transforms.functional.invert,
        transforms.RandomPerspective(p=1),
        transforms.RandomCrop(16),
        transforms.RandomAffine(degrees=90, translate=(.25, .25), scale=(.5, 1.)),
        transforms.GaussianBlur((5, 5)),
        transforms.Compose([
            transforms.functional.invert, 
            transforms.RandomPerspective(p=1)]
        ),
    ]
    for ax, t in zip(axes.flat, transformations):
        _mnist = datasets.MNIST("~/.pytorch", transform=t)
        ax.imshow(_mnist[0][0], cmap='gray')
        ax.axis('off')
    
    fig.savefig('mnist_samples_augmented.png')



if __name__ == '__main__':
    from torchvision import datasets
    from torchvision import transforms
    
    # mnist = datasets.MNIST("~/.pytorch")
    # x_raw = mnist.data.view(-1, 1, 28, 28).numpy()
    # generate_plots(x_raw, name='mnist')

    import torch
    from torch.utils.data import DataLoader, Subset, random_split

    max_epochs = 10
    debug = False

    def accuracy(pred, y):
        correct = (pred.argmax(1) == y).float()
        return correct.mean(0)

    mnist = datasets.MNIST("~/.pytorch", transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.1307, ), (.3081, ))
    ]))
    my_mnist = Subset(mnist, range(1000))
    if debug:
        mnist_train = Subset(my_mnist, [0, 1])
        mnist_valid = Subset(my_mnist, range(2, len(my_mnist)))
    else:
        mnist_train, mnist_valid = random_split(my_mnist, [800, 200])

    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, 5),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(8 * 24 * 24, 10),
    )
    ce = torch.nn.CrossEntropyLoss()
    metrics = {'loss': ce, 'acc': accuracy}

    lr=1e-2  # for lr in (5e-1, 1e-1, 1e-3):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for batch_size, max_epochs in zip((16, 64, 256), (4, 16, 64)):
        torch.manual_seed(1234)
        train_samples = DataLoader(mnist_train, batch_size=batch_size)
        valid_samples = DataLoader(mnist_valid, batch_size=len(mnist_valid))

        model[0].reset_parameters()
        model[-1].reset_parameters()
        sgd = torch.optim.SGD(model.parameters(), lr=lr)

        train_errs, valid_errs = [], []
        results = evaluate(model, valid_samples, metrics)
        valid_errs.extend(results['loss'])
        print("epoch 00", "?")
        for epoch in range(1, 1 + max_epochs):
            errs = update(model, train_samples, ce, sgd)
            train_errs.extend(errs)
            results = evaluate(model, valid_samples, metrics)
            valid_errs.extend(results['loss'])
            print(f"epoch {epoch:02d}", errs[-1])
        
        # p = plt.plot(np.linspace(0, max_epochs, len(train_errs)), train_errs, label=f'train (lr={lr:.0e})')
        # plt.plot(valid_errs, color=p[-1].get_color(), linestyle='--', label='valid')
        # plt.ylim(0, 3)
        p = ax1.plot(train_errs, label=f'train (bs={batch_size:d})')
        ax1.plot(np.linspace(0, len(train_errs), len(valid_errs)), valid_errs, 
                 color=p[-1].get_color(), linestyle='--')
        ax1.set_xlabel('updates')
        ax1.xaxis.set_ticks_position('top')
        ax1.xaxis.set_label_position('top')
        ax1.set_ylim(0, 2)
        p = ax2.plot(np.linspace(0, max_epochs, len(train_errs)), train_errs, label=f'train (bs={batch_size:d})')
        ax2.plot(valid_errs, color=p[-1].get_color(), linestyle='--')
        ax2.set_xlabel('epochs')
        ax2.set_ylim(0, 2)
    
    ax2.legend()
    # plt.savefig('lr_importance.png')
    plt.savefig('updates_vs_epochs.png', dpi=200)


    # # overfit 2 samples
    # fc = torch.nn.Sequential(
    #     torch.nn.Flatten(),
    #     torch.nn.Linear(784, 10)
    # )
    # loader = torch.utils.data.DataLoader(mnist, batch_size=2)
    # x, y = next(iter(loader))
    # valid_data = torch.utils.data.Subset(mnist, range(2, len(mnist)))
    # valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=998)
    # x_val, y_val = next(iter(valid_loader))
    # opt = torch.optim.SGD(fc.parameters(), lr=1e-1)
    # ce = torch.nn.CrossEntropyLoss()
    # errs, val_errs = [], []
    # for _ in range(1, 6):
    #     pred = fc(x)
    #     err = ce(pred, y)
    #     errs.append(err.item())

    #     with torch.no_grad():
    #         pred = fc(x_val)
    #         val_err = ce(pred, y_val)
    #         val_errs.append(val_err.item())

    #     opt.zero_grad()
    #     err.backward()
    #     opt.step()
    
    # plt.plot(errs, label="train")
    # plt.plot(val_errs, label="valid")
    # plt.xlabel("epochs")
    # plt.ylabel("cross-entropy")
    # plt.legend()
    # plt.savefig("debug_overfit.png")

    # cifar = datasets.CIFAR10("~/.pytorch")
    # x_raw = np.moveaxis(cifar.data, -1, 1)
    # generate_plots(x_raw, name='cifar')