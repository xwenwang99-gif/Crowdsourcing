import numpy as np
from scipy.special import logsumexp


class DawidSkeneModel:
    def __init__(
        self,
        class_num,
        max_iter=100,
        tolerance=1e-6,
        smoothing=1e-2,
    ):
        self.class_num = class_num
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.smoothing = smoothing

    def run(self, dataset):
        self.dataset_tensor = np.asarray(dataset, dtype=np.float64)

        self.task_num, self.worker_num, observed_class_num = (
            self.dataset_tensor.shape
        )

        if observed_class_num != self.class_num:
            raise ValueError(
                f"Dataset has {observed_class_num} classes, "
                f"but class_num={self.class_num}."
            )

        # Majority-vote initialization with smoothing.
        votes = self.dataset_tensor.sum(axis=1)  # task x observed class
        vote_totals = votes.sum(axis=1, keepdims=True)

        predict_label = (
            votes + self.smoothing
        ) / (
            vote_totals + self.smoothing * self.class_num
        )

        for iter_num in range(self.max_iter):
            error_rates = self._m_step(predict_label)
            next_predict_label = self._e_step(
                predict_label,
                error_rates,
            )

            posterior_diff = np.max(
                np.abs(next_predict_label - predict_label)
            )

            predict_label = next_predict_label

            if posterior_diff < self.tolerance:
                break

        # Recompute parameters using the final task posteriors.
        error_rates = self._m_step(predict_label)

        marginal_predict = (
            predict_label.sum(axis=0) + self.smoothing
        ) / (
            self.task_num + self.smoothing * self.class_num
        )

        worker_reliability = {
            worker: np.dot(
                marginal_predict,
                np.diag(error_rates[worker]),
            )
            for worker in range(self.worker_num)
        }

        return (
            marginal_predict,
            error_rates,
            worker_reliability,
            predict_label,
        )

    def _m_step(self, predict_label):
        """
        counts[w, c, l] =
            expected number of times worker w outputs l
            when the true class is c.
        """
        counts = np.einsum(
            "ic,iwl->wcl",
            predict_label,
            self.dataset_tensor,
        )

        # Dirichlet smoothing prevents zero confusion probabilities.
        counts += self.smoothing

        error_rates = counts / counts.sum(
            axis=2,
            keepdims=True,
        )

        return error_rates

    def _e_step(self, predict_label, error_rates):
        marginal_probability = (
            predict_label.sum(axis=0) + self.smoothing
        ) / (
            self.task_num + self.smoothing * self.class_num
        )

        # log_error_rates[w, true_class, observed_class]
        log_error_rates = np.log(
            np.clip(error_rates, 1e-300, 1.0)
        )

        # log_likelihood[i, c]
        # = sum_w sum_l n[i,w,l] log(pi[w,c,l])
        log_likelihood = np.einsum(
            "iwl,wcl->ic",
            self.dataset_tensor,
            log_error_rates,
        )

        log_joint = (
            log_likelihood
            + np.log(marginal_probability)[None, :]
        )

        # Stable posterior normalization.
        log_posterior = log_joint - logsumexp(
            log_joint,
            axis=1,
            keepdims=True,
        )

        return np.exp(log_posterior)

    def _get_likelihood(self, predict_label, error_rates):
        marginal_probability = (
            predict_label.sum(axis=0) + self.smoothing
        ) / (
            self.task_num + self.smoothing * self.class_num
        )

        log_error_rates = np.log(
            np.clip(error_rates, 1e-300, 1.0)
        )

        log_likelihood = np.einsum(
            "iwl,wcl->ic",
            self.dataset_tensor,
            log_error_rates,
        )

        log_joint = (
            log_likelihood
            + np.log(marginal_probability)[None, :]
        )

        return np.sum(logsumexp(log_joint, axis=1))