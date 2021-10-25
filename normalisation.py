import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split
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
        Array like `x`, but shifted to have zero mean.
    """
    x = np.atleast_2d(x)
    if axis is None:
        axis = (0, *range(-1, -1 - sig_dims, -1))
    
    return x - np.mean(x, axis, keepdims=True)


def normalised(x: np.ndarray, sig_dims: int = 0, eps: float = 1e-5):
    """
    Get a normalised version of a given dataset.

    Parameters
    ----------
    x : np.ndarray
        Array with at least 2 dimensions holding the dataset to be centred.
        First dimension is assumed to be the number of samples.
    sig_dims : int, optional
        Number of dimensions of the signal in the dataset.
        This allows to compute the statistics over the entire signal,
        rather than over each entry in the signal individually.
    eps : float, optional
        Numerical constant that is used to avoid division by zero errors.

    Returns
    -------
    x_n : np.ndarray
        Array like `x`, but shifted to have zero mean 
        and scaled to have unit variance.
    """
    axis = (0, *range(-1, -1 - sig_dims, -1))
    x_c = centred(x, axis=axis)
    var = np.mean(x_c * x_c, axis, keepdims=True)
    return x_c / (var + eps) ** .5

def _whitened_flat(x: np.ndarray, zca: bool = False, eps: float = 1e-5):
    """ Get a whitened version of a dataset with scalar features. """
    _x_c = centred(x, sig_dims=0)
    x_c = _x_c.reshape(len(_x_c), -1)
    cov = x_c.T @ x_c / len(x_c)
    l, u = np.linalg.eigh(cov)
    x_rot = x_c @ u

    x_w = x_rot / np.sqrt(l + eps)
    if zca:
        x_w = x_w @ u.T
    
    return x_w.reshape(_x_c.shape)

def whitened(x: np.ndarray, zca: bool = False, sig_dims: int = 0, eps: float = 1e-5):
    """ 
    Get a whitened version of a dataset.

    Parameters
    ----------
    x : np.ndarray
        Array with at least 2 dimensions holding the dataset to be centred.
        First dimension is assumed to be the number of samples.
    zca : bool, optional
        Use ZCA whitening if `True`, otherwise use PCA whitening (default).
    sig_dims : int, optional
        Number of dimensions of the signal in the dataset.
        This allows to compute the statistics over the entire signal,
        rather than over each entry in the signal individually.
    eps : float, optional
        Numerical constant that is used to avoid division by zero errors.

    Returns
    -------
    x_w : np.ndarray
        Array like `x`, but shifted to have zero mean,
        scaled to have unit variance and rotated to have zero covariance.
    """
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


def accuracy(pred, y):
    """ Compute accuracy of predictions. """
    correct = (pred.argmax(1) == y).float()
    return correct.mean(0)


@torch.no_grad()
def evaluate(model, samples, metrics):
    """ One epoch of validation. """
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
    """ One epoch of training. """
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


def train_with_sgd(model, train_data, valid_data, metrics, 
                   lr: float = 1e-2, batch_size: int = 64, max_epochs: int = 16):
    """ Train a network with SGD for a few epochs. """
    torch.manual_seed(1234)
    train_samples = DataLoader(train_data, batch_size=batch_size)
    valid_samples = DataLoader(valid_data, batch_size=len(valid_data))

    model[0].reset_parameters()
    model[-1].reset_parameters()
    sgd = torch.optim.SGD(model.parameters(), lr=lr)

    train_errs, valid_errs = [], []
    results = evaluate(model, valid_samples, metrics)
    valid_errs.extend(results['loss'])
    print("epoch 00", "?")

    loss_func = metrics['loss']
    for epoch in range(1, 1 + max_epochs):
        errs = update(model, train_samples, loss_func, sgd)
        train_errs.extend(errs)
        results = evaluate(model, valid_samples, metrics)
        valid_errs.extend(results['loss'])
        print(f"epoch {epoch:02d}", errs[-1])
    
    return train_errs, valid_errs


def train_multiple_mnist(learning_rates: tuple = (1e-2, ), batch_sizes: tuple = (64, ), 
                         epoch_maxs: tuple = (16, ), debug: bool = False):
    """ Train a fixed architecture for MNIST on multiple hyper-parameters. """
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

    for lr in learning_rates:
        for batch_size, max_epochs in zip(batch_sizes, epoch_maxs):
            train_errs, valid_errs = train_with_sgd(
                model, mnist_train, mnist_valid, metrics,
                lr=lr, batch_size=batch_size, max_epochs=max_epochs
            )
            yield train_errs, valid_errs, {'lr': lr, 'bs': batch_size, 'ep': max_epochs}


def plot_single_sample_overfit(ax=None):
    """ 
    Create plot to illustrate overfitting behaviour on few samples. 
    
    Note
    ----
    The figure from the presentation used during the recording used an even simpler model.
    """
    if ax is None:
        ax = plt.gca()
    
    for train_errs, valid_errs, _ in train_multiple_mnist(
        learning_rates=(1e-1, ), epoch_maxs=(6, ), debug=True
    ):
        ax.plot(train_errs, label="train")
        ax.plot(valid_errs, label="valid")

    ax.set_xlabel("epochs")
    ax.set_ylabel("cross-entropy")
    ax.legend()
    return ax


def plot_learning_rates(ax=None):
    """ Create plot to illustrate importance of learning rate. """
    if ax is None:
        ax = plt.gca()
    
    for train_errs, valid_errs, kwargs in train_multiple_mnist(
        learning_rates=(5e-1, 1e-1, 1e-3)
    ):
        p = ax.plot(np.linspace(0, kwargs['ep'], len(train_errs)), train_errs, 
                    label=f"train (lr={kwargs['lr']:.0e})")
        ax.plot(valid_errs, color=p[-1].get_color(), linestyle='--', label='valid')
        ax.set_ylim(0, 3)
    
    ax.legend()
    return ax


def plot_batch_sizes(fig=None):
    """ Create plot to illustrate importance of update count. """
    if fig is None:
        fig = plt.gcf()
    
    ax1, ax2 = fig.subplots(2, 1)
    for train_errs, valid_errs, kwargs in train_multiple_mnist(
        batch_sizes=(16, 64, 256), epoch_maxs=(4, 16, 64)
    ):
        p = ax1.plot(train_errs, label=f"train (bs={kwargs['bs']:d})")
        ax1.plot(np.linspace(0, len(train_errs), len(valid_errs)), valid_errs, 
                 color=p[-1].get_color(), linestyle='--')
        ax1.set_xlabel('updates')
        ax1.xaxis.set_ticks_position('top')
        ax1.xaxis.set_label_position('top')
        ax1.set_ylim(0, 2)

        p = ax2.plot(np.linspace(0, kwargs['ep'], len(train_errs)), train_errs, 
                     label=f"train (bs={kwargs['bs']:d})")
        ax2.plot(valid_errs, color=p[-1].get_color(), linestyle='--')
        ax2.set_xlabel('epochs')
        ax2.set_ylim(0, 2)
    
    ax2.legend()
    return fig


def plot_statistics(x: np.ndarray, bounds: tuple = None,
                    cmap: str = 'cividis', fig=None):
    """ 
    Visualise (pixel-wise) mean and standard deviation of image dataset.

    Parameters
    ----------
    x : np.ndarray
        Dataset to compute the statistics from.
    bounds : (float, float), optional
        Lower and upper bound for the values in the dataset.
        These bounds are important for consistent plots.
    cmap : str, optional
        The color map to use for visualising the mean.
        This can/should be the same color map as used for the original images.
    fig : plt.figure, optional
        The figure to use for plotting.

    Returns
    -------
    fig : plt.figure
        The figure with the plots.
    """
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



def plot_samples(x: np.ndarray, nrows: int = 2, ncols: int = 5,
                 bounds: tuple = None, cmap='cividis', fig=None):
    """ 
    Visualise a few samples from an image dataset.

    Parameters
    ----------
    x : np.ndarray
        Dataset to get the samples from.
    nrows : int, optional
        The number of rows with samples in the figure.
        The total number of plotted samples will be `nrows * ncols`.
    ncols : int, optional
        The number of columns with samples in the figure.
        The total number of plotted samples will be `nrows * ncols`.
    bounds : (float, float), optional
        Lower and upper bound for the values in the dataset.
        These bounds are important for consistent plots.
    cmap : str, optional
        The color map to use for visualising the mean.
        This can/should be the same color map as used for the original images.
    fig : plt.figure, optional
        The figure to use for plotting.

    Returns
    -------
    fig : plt.figure
        The figure with the plots.
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

    x_mlp_pca = whitened(x_float)
    fig_pca = plot_samples(x_mlp_pca, bounds=(-3, 3))
    fig_pca.savefig('_'.join([name, "samples_pxl_pca_clipped.png"]), **save_kwargs)
    fig_pca = plot_statistics(x_mlp_pca, bounds=(-3, 3))
    fig_pca.savefig('_'.join([name, "stats_pxl_pca_clipped.png"]), **save_kwargs)

    x_mlp_zca = whitened(x_float, zca=True)
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

    x_cnn_pca = whitened(x_float, sig_dims=2)
    fig_pca = plot_samples(x_cnn_pca)
    fig_pca.savefig('_'.join([name, "samples_img_pca.png"]), **save_kwargs)
    fig_pca = plot_statistics(x_cnn_pca)
    fig_pca.savefig('_'.join([name, "stats_img_pca.png"]), **save_kwargs)

    x_cnn_zca = whitened(x_float, zca=True, sig_dims=2)
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

    plt.figure()
    plot_single_sample_overfit()
    plt.savefig("debug_overfit.png")

    plt.figure()
    plot_learning_rates()
    plt.savefig("lr_importance.png")

    fig = plot_batch_sizes(fig=plt.figure())
    fig.savefig("updates_vs_epochs.png", dpi=200)



if __name__ == '__main__':
    mnist = datasets.MNIST("~/.pytorch")
    x_raw = mnist.data.view(-1, 1, 28, 28).numpy()
    generate_plots(x_raw, name='mnist')

    # # only useful for data pre-processing plots...
    # cifar = datasets.CIFAR10("~/.pytorch")
    # x_raw = np.moveaxis(cifar.data, -1, 1)
    # generate_plots(x_raw, name='cifar')