import matplotlib.pyplot as plt


def get_scatter_plot(y_pred, y_meas):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    ax.scatter(y_meas.cpu().numpy(), y_pred.cpu().numpy(), label="Prediction")
    ax.plot(y_meas.cpu().numpy(), y_meas.cpu().numpy(), 'k--', label="Target")
    ax.grid(True)
    ax.legend()

    return fig


def get_two_histograms(y_pred, y_meas):
    fig, ax = plt.subplots(1, 1)
    hist_range = (
        min(y_pred.cpu().numpy().min(), y_meas.cpu().numpy().min()),
        max(y_pred.cpu().numpy().max(), y_meas.cpu().numpy().max())
    )
    ax.hist(y_pred.cpu().numpy(), alpha=0.5, color='g', bins=30, range=hist_range, label="Prediction")
    ax.hist(y_meas.cpu().numpy(), alpha=0.5, color='r', bins=30, range=hist_range, label="Target")
    ax.legend()

    return fig


def get_error_histogram(y_pred, y_meas):
    fig, ax = plt.subplots(1, 1)
    ax.hist(y_pred.cpu().numpy() - y_meas.cpu().numpy(), bins=30)
    ax.axvline(x=0, color='k', linewidth=2)

    return fig
