import os
import multiprocessing as mp
import numpy as np
import torchvision

# ==========================================
# 1. FUNCIONES MATEMÁTICAS BÁSICAS
# ==========================================
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def deriv_sigmoid(A):
    return A * (1 - A)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / expZ.sum(axis=0, keepdims=True)

def one_hot(Y):
    oh = np.zeros((Y.size, 10))
    oh[np.arange(Y.size), Y] = 1
    return oh.T

def init_params():
    W1 = np.random.randn(10, 784) * 0.01
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * 0.01
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2


# ==========================================
# 2. TAREA DEL TRABAJADOR (EJECUTADA EN CADA NÚCLEO)
# ==========================================
def worker_task(worker_id, X_chunk, Y_chunk, W1, b1, W2, b2, verbose=False):
    """
    Cada proceso procesa exclusivamente su fragmento de imágenes.
    Calcula las sumas crudas (sin dividir entre m) según la Opción A.
    """
    pid = os.getpid()
    if verbose:
        print(f"[Worker {worker_id} | PID {pid}] Procesando fragmento de {X_chunk.shape[1]} muestras...")

    # Forward Propagation local
    Z1 = W1.dot(X_chunk) + b1
    A1 = sigmoid(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    # Backward Propagation (Sumas brutas)
    Y_oh = one_hot(Y_chunk)
    dZ2 = A2 - Y_oh
    
    # Nótese: NO multiplicamos por 1/m aquí
    raw_dW2 = dZ2.dot(A1.T)
    raw_db2 = np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * deriv_sigmoid(A1)
    raw_dW1 = dZ1.dot(X_chunk.T)
    raw_db1 = np.sum(dZ1, axis=1, keepdims=True)

    # Conteo de predicciones correctas en este fragmento
    predictions = np.argmax(A2, axis=0)
    correct = np.sum(predictions == Y_chunk)

    return raw_dW1, raw_db1, raw_dW2, raw_db2, correct


# ==========================================
# 3. ENTRENAMIENTO PARALELO MULTIPROCESO
# ==========================================
def train_parallel(X, Y, num_workers=4, iterations=100, alpha=0.5):
    m_total = Y.size
    chunk_size = m_total // num_workers

    # Partición de datos en fragmentos homogéneos
    X_chunks = [X[:, i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]
    Y_chunks = [Y[i * chunk_size:(i + 1) * chunk_size] for i in range(num_workers)]

    W1, b1, W2, b2 = init_params()

    print(f"\n--- Iniciando entrenamiento con {num_workers} procesos concurrentes ---")
    print(f"Total imágenes: {m_total} | Imágenes por proceso: {chunk_size}\n")

    # Mantenemos un Pool activo para no destruir y recrear procesos en cada iteración
    with mp.Pool(processes=num_workers) as pool:
        for i in range(iterations):
            # Solo imprimimos los PIDs en la primera iteración para no saturar la terminal
            verbose = (i == 0)

            # Empaquetamos los argumentos para cada trabajador
            tasks = [
                (w_id, X_chunks[w_id], Y_chunks[w_id], W1, b1, W2, b2, verbose)
                for w_id in range(num_workers)
            ]

            # starmap distribuye las tareas entre los procesos disponibles
            results = pool.starmap(worker_task, tasks)

            # --- OPCIÓN A: ACUMULACIÓN Y ESCALADO GLOBAL ---
            # 1. Sumamos las matrices crudas de todos los procesos
            total_raw_dW1 = sum(r[0] for r in results)
            total_raw_db1 = sum(r[1] for r in results)
            total_raw_dW2 = sum(r[2] for r in results)
            total_raw_db2 = sum(r[3] for r in results)
            total_correct = sum(r[4] for r in results)

            # 2. Dividimos exactamente entre el m global (50,000)
            dW1 = (1 / m_total) * total_raw_dW1
            db1 = (1 / m_total) * total_raw_db1
            dW2 = (1 / m_total) * total_raw_dW2
            db2 = (1 / m_total) * total_raw_db2

            # 3. Actualizamos los pesos centrales
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

            if i % 20 == 0 or i == iterations - 1:
                accuracy = total_correct / m_total
                print(f"Iteración {i:3d} | Precisión Global: {accuracy:.4f}")

    return W1, b1, W2, b2


# ==========================================
# 4. PUNTO DE ENTRADA PRINCIPAL
# ==========================================
if __name__ == '__main__':
    # Descarga y carga de MNIST
    print("Cargando dataset...")
    full_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    all_X = full_dataset.data.numpy().reshape(60000, 784).T / 255.0
    all_Y = full_dataset.targets.numpy()

    X_train = all_X[:, :50000]
    Y_train = all_Y[:50000]
    X_test = all_X[:, 50000:]
    Y_test = all_Y[50000:]

    # Ajusta num_workers según los núcleos lógicos que desees poner a trabajar
    NUM_WORKERS = 4
    W1, b1, W2, b2 = train_parallel(X_train, Y_train, num_workers=NUM_WORKERS, iterations=100, alpha=0.5)

    # Evaluación secuencial final sobre el conjunto de test
    Z1 = W1.dot(X_test) + b1
    A1 = sigmoid(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    test_preds = np.argmax(A2, axis=0)
    test_acc = np.sum(test_preds == Y_test) / Y_test.size
    print(f"\nPrecisión en Test: {test_acc:.4f} ({int(test_acc * len(Y_test))}/{len(Y_test)} correctas)")