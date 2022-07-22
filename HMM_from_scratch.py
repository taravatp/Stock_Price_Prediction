import numpy as np


class HMM:
    def __init__(self, num_states, observation_dim):

        self.num_states = num_states
        self.observation_dim = observation_dim
        self.A = np.random.uniform(0, 1, size=(num_states, num_states))
        self.B = np.random.uniform(0, 1, size=(num_states, observation_dim))
        self.pi = np.full((1, num_states), 1 / num_states)

    def forward(self, observation):
        num_rows = self.num_states  # number of states
        num_cols = self.observation_dim  # number of observations
        alpha_table = np.zeros((num_rows, num_cols))  # instantiating the alpha table
        alpha_table[:, 0] = self.pi * self.B[:, 0]  # filling the first column

        for col in range(1, num_cols):
            for row in range(num_rows):
                alpha_table[row, col] = np.dot(alpha_table[:, col - 1], self.A[:, row]) * self.B[row, observation[col]]

        return alpha_table

    def backward(self, observation):
        num_rows = self.num_states  # number of states
        num_cols = self.observation_dim  # number of observations
        beta_table = np.zeros((num_rows, num_cols))  # instantiating the beta table
        beta_table[:, -1] = 1  # filling the last column

        for col in reversed(range(num_cols - 1)):
            for row in range(num_rows):
                beta_table[row, col] = np.sum(beta_table[:, col + 1] * self.A[row, :] * self.B[:, observation[col + 1]])

        return beta_table

    def decode(self, observation):
        num_rows = self.num_states  # number of states
        num_cols = self.observation_dim  # number of observations
        viterbi_table = np.zeros((num_rows, num_cols))
        viterbi_table[:, 0] = self.pi * self.B[:, 0]  # filling the first column

        indexes = np.zeros((num_cols - 1, num_rows), dtype=int)

        for col in range(1, num_cols):
            for row in range(num_rows):
                probs = viterbi_table[:, col - 1] * self.A[:, row] * self.B[row, observation[col]]
                viterbi_table[row, col] = np.max(probs)  # storing the maximum value
                indexes[col - 1, row] = np.argmax(probs)  # storing the maximum index
        return viterbi_table, indexes

    def baulm_wetch_train(self, observations):
        a = np.zeros((self.num_states, self.num_states))
        b = np.zeros((self.num_states, self.B.shape[0]))

        for iteration in range(200):
            for observation in observations:
                alpha_table = self.forward(observation)
                beta_table = self.backward(observation)

                etha_table = np.zeros((self.num_states, self.num_states, len(observation) - 1))

                denominator = np.sum(alpha_table[:, -1])
                for i in range(etha_table.shape[0]):
                    for j in range(etha_table.shape[1]):
                        for t in range(etha_table.shape[2]):
                            etha_table[i, j, t] = (alpha_table[i, t] * self.A[i, j] * self.B[j, t + 1] * beta_table[
                                j, t + 1]) / denominator

                gamma_table = np.zeros((self.num_states, len(observation)))
                for i in range(gamma_table.shape[0]):
                    for j in range(gamma_table.shape[1]):
                        gamma_table[i, j] = (alpha_table[i, j] * beta_table[i, j]) / denominator

                # initializing matrix A
                for i in range(a.shape[0]):
                    for j in range(a.shape[1]):
                        numerator = sum(etha_table[i, j, :])
                        dominator = sum(gamma_table[i, :])
                        a[i, j] = numerator / dominator

                # initializing matrix B
                for i in range(b.shape[0]):
                    for j in range(b.shape[1]):
                        # farz mikonim ke observation haye ma index ha hastand - maghadire 0,1,2,... darain
                        values = np.unique(observation)
                        numerator = sum(gamma_table[i, values])  # not sure about the syntax
                        dominator = sum(gamma_table[i, :])
                        b[i, j] = numerator / dominator

            self.A = a
            self.B = b
