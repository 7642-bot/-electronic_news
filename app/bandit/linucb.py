import numpy as np

class LinUCB:
    def __init__(self, n_arms: int, context_dim: int, alpha: float = 0.5):
        self.n_arms = n_arms
        self.d = context_dim
        self.alpha = alpha
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros((context_dim, 1)) for _ in range(n_arms)]

    def _ucb(self, arm: int, x: np.ndarray) -> float:
        A_inv = np.linalg.inv(self.A[arm])
        theta = A_inv @ self.b[arm]
        exploit = float((theta.T @ x).item())          # was: float(theta.T @ x)
        explore = self.alpha * float(np.sqrt((x.T @ A_inv @ x).item()))
        return exploit + explore

    def select(self, x: np.ndarray) -> int:
        return int(np.argmax([self._ucb(a, x) for a in range(self.n_arms)]))

    def update(self, arm: int, x: np.ndarray, reward: float):
        self.A[arm] += x @ x.T
        self.b[arm] += reward * x
