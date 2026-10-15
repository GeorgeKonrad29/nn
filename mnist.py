# /// script
# dependencies = [
#     "marimo>=0.23.3",
#     "matplotlib==3.10.8",
#     "numpy==2.4.2",
#     "torch==2.10.0",
#     "torchvision==0.25.0",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The MNIST dataset

    First, we will load the MNIST dataset. The following slider will allow you to explore it.

    - There are 60.000 images in the dataset.
    - We will use 50.000 to train the neural network
    - and 10.000 to test its accuracy
    """)
    return


@app.cell
def _(mo, torchvision):
    full_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True
    )
    all_X = full_dataset.data.numpy().reshape(60000, 784).T / 255.0
    all_Y = full_dataset.targets.numpy()

    X_train = all_X[:, :50000]
    Y_train = all_Y[:50000]

    X_test = all_X[:, 50000:]
    Y_test = all_Y[50000:]

    mo.md(f"Training set size: {X_train.shape[1]} | Test set size: {X_test.shape[1]}")
    return X_test, X_train, Y_test, Y_train, full_dataset


@app.cell(hide_code=True)
def _(mo):
    index_slider = mo.ui.slider(0, 60000-1, label="Select Image Index")
    index_slider
    return (index_slider,)


@app.cell(hide_code=True)
def _(full_dataset, index_slider, mo, plt):
    raw_images = full_dataset.data
    raw_labels = full_dataset.targets

    raw_img = raw_images[index_slider.value]
    raw_label = raw_labels[index_slider.value]

    plt.imshow(raw_img, cmap='gray')
    plt.title(f"Expected output: {raw_label.item()}")
    plt.axis('off')
    mo.as_html(plt.gcf())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameter initialization
    """)
    return


@app.cell
def _(np):
    def init_params():
        W1 = np.random.randn(10, 784) * 0.01
        b1 = np.zeros((10, 1))
        W2 = np.random.randn(10, 10) * 0.01
        b2 = np.zeros((10, 1))
        return W1, b1, W2, b2

    return (init_params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helper functions
    """)
    return


@app.cell
def _(np):
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    def deriv_sigmoid(A):
        return A * (1 - A)

    def softmax(Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def one_hot(Y):
        oh = np.zeros((Y.size, 10))
        oh[np.arange(Y.size), Y] = 1
        return oh.T

    return deriv_sigmoid, one_hot, sigmoid, softmax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Forward evaluation
    """)
    return


@app.cell
def _(sigmoid, softmax):
    def forward_prop(W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = sigmoid(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2

    return (forward_prop,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Backward propagation
    """)
    return


@app.cell
def _(deriv_sigmoid, np, one_hot):
    def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
        m = Y.size
        Y_oh = one_hot(Y)

        dZ2 = A2 - Y_oh
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = W2.T.dot(dZ2) * deriv_sigmoid(A1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parameter update
    """)
    return


@app.function
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Network training
    """)
    return


@app.cell
def _(np):
    def worker_process(conn, X_chunk, Y_chunk):
        """Bucle continuo ejecutado en un proceso CPU independiente."""
        while True:
            msg = conn.recv()
            if msg is None:
                break

            W1, b1, W2, b2 = msg

            # 1. Forward propagation local
            Z1 = W1.dot(X_chunk) + b1
            A1 = 1 / (1 + np.exp(-Z1))
            Z2 = W2.dot(A1) + b2
            expZ = np.exp(Z2 - np.max(Z2, axis=0, keepdims=True))
            A2 = expZ / expZ.sum(axis=0, keepdims=True)

            # 2. Backward propagation (Sumas de la Opción A)
            Y_oh = np.zeros((Y_chunk.size, 10))
            Y_oh[np.arange(Y_chunk.size), Y_chunk] = 1
            Y_oh = Y_oh.T

            dZ2 = A2 - Y_oh
            S_W2 = dZ2.dot(A1.T)
            S_b2 = np.sum(dZ2, axis=1, keepdims=True)

            dZ1 = W2.T.dot(dZ2) * (A1 * (1 - A1))
            S_W1 = dZ1.dot(X_chunk.T)
            S_b1 = np.sum(dZ1, axis=1, keepdims=True)

            preds = np.argmax(A2, axis=0)
            correct_count = int(np.sum(preds == Y_chunk))

            conn.send((S_W1, S_b1, S_W2, S_b2, correct_count))

    return (worker_process,)


@app.cell
def _(init_params, np, worker_process):
    import multiprocessing as mp

    def train_model_parallel(X, Y, iterations=100, alpha=0.5, num_workers=4):
        m_total = Y.size
        W1, b1, W2, b2 = init_params()
        history = []

        X_chunks = np.array_split(X, num_workers, axis=1)
        Y_chunks = np.array_split(Y, num_workers)

        ctx = mp.get_context("fork")
        processes = []
        parent_conns = []

        for rank in range(num_workers):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=worker_process,
                args=(child_conn, X_chunks[rank], Y_chunks[rank]),
                daemon=True,
            )
            p.start()
            processes.append(p)
            parent_conns.append(parent_conn)

        try:
            for i in range(iterations):
                for conn in parent_conns:
                    conn.send((W1, b1, W2, b2))

                sum_W1 = np.zeros_like(W1)
                sum_b1 = np.zeros_like(b1)
                sum_W2 = np.zeros_like(W2)
                sum_b2 = np.zeros_like(b2)
                total_correct = 0

                for conn in parent_conns:
                    s_w1, s_b1, s_w2, s_b2, correct = conn.recv()
                    sum_W1 += s_w1
                    sum_b1 += s_b1
                    sum_W2 += s_w2
                    sum_b2 += s_b2
                    total_correct += correct

                dW1 = sum_W1 / m_total
                db1 = sum_b1 / m_total
                dW2 = sum_W2 / m_total
                db2 = sum_b2 / m_total

                W1, b1, W2, b2 = update_params(
                    W1, b1, W2, b2, dW1, db1, dW2, db2, alpha
                )

                if i % 10 == 0 or i == iterations - 1:
                    accuracy = total_correct / m_total
                    history.append(accuracy)
                    print(f"Iteración {i:3d} | Precisión: {accuracy:.4f}")

        finally:
            for conn in parent_conns:
                conn.send(None)
            for p in processes:
                p.join()

        return W1, b1, W2, b2, history

    return (train_model_parallel,)


@app.cell
def _(X_train, Y_train, train_model_parallel):
    params = train_model_parallel(
        X_train, Y_train, iterations=100, alpha=0.5, num_workers=4
    )
    W1, b1, W2, b2, stats = params
    return W1, W2, b1, b2, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model evaluation
    """)
    return


@app.cell
def _(W1, W2, X_test, Y_test, b1, b2, forward_prop, mo, np):
    _, _, _, A2_test = forward_prop(W1, b1, W2, b2, X_test)
    test_predictions = np.argmax(A2_test, axis=0)
    test_accuracy = np.sum(test_predictions == Y_test) / Y_test.size
    mo.md(f"**Test accuracy: {test_accuracy:.4f}** ({int(test_accuracy * len(Y_test))}/{len(Y_test)} correct)")
    return


@app.cell
def _(mo, plt, stats):
    fig, ax = plt.subplots()
    ax.plot(range(0, len(stats) * 50, 50), stats, marker='o')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training accuracy")
    ax.set_title("Training accuracy over time")
    ax.set_ylim(0, 1)
    ax.grid(True)
    mo.as_html(fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prediction demo
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    demo_slider = mo.ui.slider(0, 60000-1, label="Select Image Index")
    demo_slider
    return (demo_slider,)


@app.cell(hide_code=True)
def _(W1, W2, b1, b2, demo_slider, forward_prop, full_dataset, mo, np, plt):
    demo_img = full_dataset.data[demo_slider.value].numpy()
    demo_label = full_dataset.targets[demo_slider.value].item()
    demo_x = demo_img.reshape(784, 1) / 255.0

    _, _, _, demo_A2 = forward_prop(W1, b1, W2, b2, demo_x)
    demo_pred = int(np.argmax(demo_A2))
    demo_conf = float(demo_A2[demo_pred, 0])

    fig2, axes = plt.subplots(1, 2, figsize=(8, 3))

    axes[0].imshow(demo_img, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title(f"Label: {demo_label} | Predicted: {demo_pred}")

    axes[1].bar(range(10), demo_A2[:, 0])
    axes[1].set_xticks(range(10))
    axes[1].set_xlabel("Digit")
    axes[1].set_ylabel("Probability")
    axes[1].set_title(f"Confidence: {demo_conf:.2%}")

    mo.as_html(fig2)
    return


@app.cell
def _():
    import marimo as mo
    import torch
    import torchvision
    import matplotlib.pyplot as plt
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return mo, np, plt, torchvision


if __name__ == "__main__":
    app.run()
