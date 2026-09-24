from src.dawid_skene_model import DawidSkeneModel
from operator import mul
import numpy as np
from scipy import stats
import numpy_indexed as npi
from sklearn.cluster import KMeans
from itertools import permutations
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class LFGP_PAPER():

    """
    Latent Factor modeling with Grouping Penalty
    """

    def __init__(self, lf_dim=3, n_worker_group=3, lambda1=1, lambda2=1):

        # Specify hyper-parameters

        self.lf_dim = lf_dim                    # dimension of latent factors
        self.n_worker_group= n_worker_group     # number of worker subgroups
        self.lambda1 = lambda1                  # penalty coefficient for task subgrouping
        self.lambda2 = lambda2                  # penalty coefficient for worker subgrouping

    def _prescreen(self, data):

        # fetch information from crowdsourced data
        n_task = len(np.unique(data[:, 0]))         # number of tasks
        n_worker = len(np.unique(data[:, 1]))       # number of workers
        n_task_group = len(np.unique(data[:, 2]))   # number of task categories
        n_record = len(data[:, 0])                  # number of crowdsourced labels

        self.n_task = n_task
        self.n_worker = n_worker
        self.n_task_group = n_task_group  
        self.n_record = n_record

    def _init_task_member_mv(self, data):

        # initialize model parameters (task initial subgroup membership) using majority voting scheme

        task = np.unique(data[:, 0]) # task index
        label = np.zeros((self.n_task, 2))
        label[:, 0] = task

        for i in range(self.n_task):
            val, _ = stats.mode(data[data[:, 0] == task[i], 2]) # get the majority of crowdsourced label for each task
            label[i, 1] = val[0]                                # assign the majority voted label to the initial label

        return label

    def _init_task_member_ds(self, data):

        data_tensor = self.data_converter(data)
        model = DawidSkeneModel(self.n_task_group, max_iter=50, tolerance=10e-100)
        _, _, _, pred_label = model.run(data_tensor)

        label = np.zeros((self.n_task, 2))
        task = np.unique(data[:, 0]) # task index
        label[:, 0] = task
        label[:, 1] = np.argmax(pred_label, axis=1).squeeze()
        

        return label

    def _init_task_member_random(self, data):

        # initialize model parameters (task initial subgroup membership) using random scheme

        task = np.unique(data[:, 0])
        label = np.zeros((self.n_task, 2))
        label[:, 0] = task

        label[:, 1] = np.random.randint(low = 0, high = self.n_task_group, size = (self.n_task, ))

        return label

    def _init_worker_member_acc(self, data, label):

        # initialize model parameters (worker initial subgroup membership) by truncating the surrogate accuracy

        worker = np.unique(data[:, 1]) # worker index
        acc = np.zeros((self.n_worker, 2))
        acc[:, 0] = worker
        member = np.zeros((self.n_worker, 2))
        member[:, 0] = worker

        for i in range(self.n_worker):
            crowd_w = data[data[:, 1] == worker[i], :]
            task_w = crowd_w[:, 0]
            acc[i, 1] = sum(label[np.isin(label[:, 0], task_w), 1] == crowd_w[:, 2]) / crowd_w.shape[0]
        
        try:

            member[:, 1] = np.digitize(acc[:, 1], [np.min(acc[:, 1]) + (np.max(acc[:, 1]) + 0.01 - np.min(acc[:, 1])) * i / self.n_worker_group for i in range(1, self.n_worker_group + 1)], right=True)

        except:

            member[:, 1] = np.digitize(acc[:, 1], [np.quantile(acc[:, 1], i/self.n_worker_group) for i in range(1, self.n_worker_group + 1)], right=True)

        return member

    def _init_worker_member_random(self, data):

        # initialize model parameters (worker initial subgroup membership) using random scheme
    
        worker = np.unique(data[:, 1])
        member = np.zeros((self.n_worker, 2))
        member[:, 0] = worker

        member[:, 1] = np.random.randint(low = 0, high=self.n_worker_group, size = (self.n_worker, ))

        return member

    def _init_task_lf_gp(self, label):

        # initialize model parameters (task latent factors) using surrogate group information

        lf = np.zeros((self.n_task, self.lf_dim))
        for i in range(self.n_task_group):

            task_idx = label[label[:, 1] == i, 0].astype(int)
            tmp_centroid = 2 * np.random.rand(self.lf_dim) - 1
            tmp_centroid = tmp_centroid / np.linalg.norm(tmp_centroid)
            lf[task_idx, :] = np.random.multivariate_normal(tmp_centroid, 0.2 * np.eye(self.lf_dim), len(task_idx))

        return lf

    def _init_worker_lf_gp(self, member):

        # initialize model parameters (worker latent factors) using surrogate group information

        lf = np.zeros((self.n_worker, self.lf_dim))

        for i in range(self.n_worker_group):

            worker_idx = member[member[:, 1] == i, 0].astype(int)
            tmp_centroid = 2 * np.random.rand(self.lf_dim) - 1
            tmp_centroid = tmp_centroid / np.linalg.norm(tmp_centroid)
            lf[worker_idx, :] = np.random.multivariate_normal(tmp_centroid, 0.2 * np.eye(self.lf_dim), len(worker_idx))

        return lf

    def _init_task_lf_random(self):

        lf = np.random.multivariate_normal(np.zeros((self.lf_dim, )), np.eye(self.lf_dim), self.n_task)

        return lf

    def _init_worker_lf_random(self):

        lf = np.random.multivariate_normal(np.zeros((self.lf_dim, )), np.eye(self.lf_dim), self.n_worker)

        return lf

    def _init_binary_params(self, data, scheme = "mv"):

        # initialize model parameters for binary crowdsourcing
        # two initialization schemes are available: mv and random

        if scheme == "mv":

            task_member = self._init_task_member_mv(data)
            worker_member = self._init_worker_member_acc(data, task_member)

            U = task_member[:, 1]
            V = worker_member[:, 1]

            A = self._init_task_lf_gp(task_member)
            B = self._init_worker_lf_gp(worker_member)

        if scheme == "ds":

            task_member = self._init_task_member_ds(data)
            worker_member = self._init_worker_member_acc(data, task_member)

            U = task_member[:, 1]
            V = worker_member[:, 1]

            A = self._init_task_lf_gp(task_member)
            B = self._init_worker_lf_gp(worker_member)

        elif scheme == "random":
            
            task_member = self._init_task_member_random(data)
            worker_member = self._init_worker_member_random(data)

            U = task_member[:, 1]
            V = worker_member[:, 1]

            A = self._init_task_lf_random()
            B = self._init_worker_lf_random()

        self.A, self.B = A, B
        self.U, self.V = U, V

    def _init_orth_iden(self):

        # initialize orthogonal matrices for multi-categorical crowdsourcing with identity matrices

        O = np.zeros((self.n_task_group, self.n_worker_group, self.lf_dim, self.lf_dim))

        for i in range(self.n_task_group):

            for j in range(self.n_worker_group):

                O[i, j, :, :] = np.eye(self.lf_dim)

        return O

    def _init_orth_rand(self):

        # initialize orthogonal matrices for multi-categorical crowdsourcing with random orthogonal matrices

        O = np.zeros((self.n_task_group, self.n_worker_group, self.lf_dim, self.lf_dim))

        for i in range(self.n_task_group):
    
            for j in range(self.n_worker_group):
                
                if i == 0:
                    O[i, j, :, :] = np.eye(self.lf_dim)
                else:
                    O[i, j, :, :] = stats.ortho_group.rvs(self.lf_dim)
        return O

    def _init_mc_params(self, data, scheme="mv"):

        # initialize model parameters for multicategory crowdsourcing
        # two initialization schemes are available: mv and random

        if scheme == "mv":

            task_member = self._init_task_member_mv(data)
            worker_member = self._init_worker_member_acc(data, task_member)

            U = task_member[:, 1]
            V = worker_member[:, 1]

            if len(np.unique(V)) < self.n_worker_group:
                worker_member = self._init_worker_member_random(data)
                V = worker_member[:, 1]

            A = self._init_task_lf_gp(task_member)
            B = self._init_worker_lf_gp(worker_member)
            O = self._init_orth_rand()

        elif scheme == "ds":

            task_member = self._init_task_member_ds(data)
            worker_member = self._init_worker_member_acc(data, task_member)

            U = task_member[:, 1]
            V = worker_member[:, 1]

            if len(np.unique(V)) < self.n_worker_group:
                worker_member = self._init_worker_member_random(data)
                V = worker_member[:, 1]

            A = self._init_task_lf_gp(task_member)
            B = self._init_worker_lf_gp(worker_member)
            O = self._init_orth_rand()

        elif scheme == "random":

            task_member = self._init_task_member_random(data)
            worker_member = self._init_worker_member_random(data)

            U = task_member[:, 1]
            V = worker_member[:, 1]

            A = self._init_task_lf_random()
            B = self._init_worker_lf_random()
            O = self._init_orth_rand()

        U = U.astype(int)
        V = V.astype(int)

        self.A, self.B, self.O = A, B, O
        self.U, self.V = U, V

    @staticmethod
    def _sigmoid(x):

        z = []

        for xi in x:

            if xi < 0:

                z.append(np.exp(xi) / (1 + np.exp(xi)))

            else:

                z.append(1.0 / (1.0 + np.exp(-xi)))

        return np.array(z)

    @staticmethod
    def _softmax(x):

        return np.divide(np.exp(x), np.sum(np.exp(x)))

    def _comp_centroid(self, A, B, U, V):

        group_A, centroid_A = npi.group_by(U).mean(A, axis=0)
        group_B, centroid_B = npi.group_by(V).mean(B, axis=0)

        Centroid_A = np.zeros(A.shape)
        Centroid_B = np.zeros(B.shape)

        for g in range(self.n_task_group):
            Centroid_A[U == group_A[g], :] = centroid_A[g, :]
        for g in range(self.n_worker_group):
            Centroid_B[V == group_B[g], :] = centroid_B[g, :]

        return Centroid_A, Centroid_B

    def _binary_loss_func(self, data):
    
        _, task_id = np.unique(data[:, 0], return_inverse=True)
        _, worker_id = np.unique(data[:, 1], return_inverse=True)

        loss =  - np.sum(data[:, 2] * np.log(self._sigmoid(np.sum(self.A[task_id, :] * self.B[worker_id, :], axis=1)))) - \
                   np.sum((1 - data[:, 2]) * np.log(self._sigmoid(-np.sum(self.A[task_id, :] * self.B[worker_id, :], axis=1))))

        gA, alpha = npi.group_by(self.U).mean(self.A, axis=0)
        gB, beta = npi.group_by(self.V).mean(self.B, axis=0)

        penalty1 = self.lambda1 * np.sum([np.linalg.norm(self.A[self.U == u, :] - alpha[gA == u, :]) ** 2 for u in np.unique(self.U)])
        penalty2 = self.lambda2 * np.sum([np.linalg.norm(self.B[self.V == v, :] - beta[gB == v, :]) ** 2 for v in np.unique(self.V)])

        return loss + penalty1 + penalty2

    def _mc_loss_func(self, data):

        _, task_id = np.unique(data[:, 0], return_inverse=True)
        _, worker_id = np.unique(data[:, 1], return_inverse=True)

        loss = 0

        for i in range(self.n_record):
        
            prob = [self.A[task_id[i], :].dot(self.O[cla, self.V[worker_id[i]], :, :]).dot(self.B[worker_id[i], :]) for cla in np.arange(self.n_task_group)]
            prob = self._softmax(prob)

            loss += - np.log(prob[int(data[i, 2])])

        gA, alpha = npi.group_by(self.U).mean(self.A, axis=0)
        gB, beta = npi.group_by(self.V).mean(self.B, axis=0)

        penalty1 = self.lambda1 * np.sum([np.linalg.norm(self.A[self.U == p, :] - alpha[gA == p, :]) ** 2 for p in np.unique(self.U)])
        penalty2 = self.lambda2 * np.sum([np.linalg.norm(self.B[self.V == q, :] - beta[gB == q, :]) ** 2 for q in np.unique(self.V)])

        return loss + penalty1 + penalty2

    def _binary_fit(self, data, scheme="mv", maxiter=100, epsilon=1e-3, verbose=0):

        task, task_id = np.unique(data[:, 0], return_inverse=True)
        worker, worker_id = np.unique(data[:, 1], return_inverse=True)

        self._init_binary_params(data, scheme=scheme)

        iter = 0
        loss_cur = self._binary_loss_func(data)
        err = 1

        if verbose > 0:
            print("Iter: {0}, loss: {1}".format(iter, loss_cur))

        A_cur, B_cur = self.A, self.B
        U_cur, V_cur = self.U, self.V

        while err > epsilon:

            A_prev, B_prev = A_cur, B_cur
            U_prev, V_prev = U_cur, V_cur
            Alpha, Beta = self._comp_centroid(A_prev, B_prev, U_prev, V_prev)

            loss_prev = loss_cur

            iter = iter + 1

            # update A
            for id, t in enumerate(task):

                obsIdx = np.argwhere(task_id == id).squeeze() # task i crowd answers
                obsWorker = worker_id[obsIdx]           # workers rated task i
                obsB = B_prev[obsWorker, :]       # squeeze if only 1 worker rated task i
                obsY = data[obsIdx, 2]
                if obsIdx.size < 2:
                    obsB = obsB[np.newaxis, :]
                    obsY = np.array([obsY])

                A_cur[id, :] = self.logistic_reg(obsB, obsY, self.lambda1, Alpha[id, :].T)

            # update B
            for jd, w in enumerate(worker):

                obsIdx = np.argwhere(worker_id == jd).squeeze() # worker j crowd answers
                obsTask = task_id[obsIdx]           # tasks rated by worker j
                obsA = A_prev[obsTask, :]       # squeeze if worker j only rated 1 task
                obsY = data[obsIdx, 2]
                if obsIdx.size < 2:
                    obsA = obsA[np.newaxis, :]
                    obsY = np.array([obsY])

                B_cur[jd, :] = self.logistic_reg(obsA, obsY, self.lambda2, Beta[jd, :].T)

            # update U
            U_cur = KMeans(n_clusters=self.n_task_group).fit_predict(A_cur)
            U_cur = self.label_swap(U_cur, U_prev)

            # update V
            V_cur = KMeans(n_clusters=self.n_worker_group).fit_predict(B_cur)
            V_cur = self.label_swap(V_cur, V_prev)

            self.A = A_cur
            self.B = B_cur
            self.U = U_cur
            self.V = V_cur

            loss_cur = self._binary_loss_func(data)

            if verbose > 0:
                print("Iter: {0}, loss: {1}".format(iter, loss_cur))

            err = np.abs(loss_cur - loss_prev) / loss_prev

            if iter > maxiter:
                break

            self.iter = iter

        self.A, self.B, self.U, self.V = A_cur, B_cur, U_cur, V_cur

    def logistic_reg(self, X, y, lambda1, centroid):
    
        # logistic regression for updating latent factors in binary crowdsourcing
        # penalty is to shrink beta to a given centroid
        if X.ndim < 2:
            X = np.expand_dims(X, axis=0)

        n, p = X.shape
        beta = np.zeros((p, ))
        y_hat = self.sigmoid(X.dot(beta)).squeeze()
        y_hat[y_hat < 1e-8] = 1e-8
        y_hat[y_hat > 1 - 1e-8] = 1 - 1e-8
        grad = X.T.dot(y_hat - y) + 2 * lambda1 * (beta - centroid)
        hessian = X.T.dot(np.diagflat(np.multiply(y_hat, 1 - y_hat))).dot(X) + 2 * lambda1 * np.eye(p)
        iter = 0
        while np.linalg.norm(grad) > 1e-5:
            beta = beta - np.linalg.inv(hessian).dot(grad)
            y_hat = self.sigmoid(X.dot(beta))
            y_hat[y_hat < 1e-8] = 1e-8
            y_hat[y_hat > 1 - 1e-8] = 1 - 1e-8
            grad = X.T.dot(y_hat - y) + 2 * lambda1 * (beta - centroid)
            hessian = X.T.dot(np.diagflat(np.multiply(y_hat, 1 - y_hat))).dot(X) + 2 * lambda1 * np.eye(p)
            iter = iter + 1
            if iter > 100:
                break
        return beta.T

    def sigmoid(self, x):

        return np.divide(1, (1 + np.exp(-x)))


    def label_swap(self, Grp_cur, Grp_prev):

        grp = np.unique(Grp_cur)

        perm_all = list(permutations(grp))

        rand_index_list = np.zeros((len(perm_all), ))

        for i in range(len(perm_all)):

            dic = {idx: perm_all[i][idx] for idx in grp}
            Grp_perm = [dic[Grp_cur[j]] for j in range(len(Grp_cur))]

            rand_index_list[i] = accuracy_score(Grp_prev, Grp_perm)

        ix = np.argmax(rand_index_list)
        dic = {idx: perm_all[ix][idx] for idx in grp}
        Grp_perm = [dic[Grp_cur[j]] for j in range(len(Grp_cur))]

        return np.array(Grp_perm)

    def data_converter(self, data):

        n_task = len(np.unique(data[:, 0]))
        n_worker = len(np.unique(data[:, 1]))
        num_class = len(np.unique(data[:, 2]))

        data_tensor = np.zeros((n_task, n_worker, num_class))

        for row in data:

            data_tensor[int(row[0]), int(row[1]), int(row[2])] += 1

        return data_tensor


    def multinomial_reg1(self, A_init, B, Y, O, lambd, centroid):

        N = B.shape[0]
        k = B.shape[1]
        C = O.shape[0]

        conc1 = np.zeros((N, C))
        conc2 = np.zeros((N, C, k))
        beta = A_init

        for n in range(N):

            for c in range(C):
                
                conc1[n, c] = np.dot(np.dot(beta.T, O[c, n, :, :]), B[n, :])
                conc2[n, c, :] = np.dot(O[c, n, :, :], B[n, :])

        conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1)[:, np.newaxis]

        grad = np.sum(- conc2[np.arange(N), Y.astype('int'), :] + \
                np.sum(conc1[:, :, np.newaxis] * conc2, axis=1), axis=0) + \
                2 * lambd * (beta - centroid)
        
        iter = 0
        while np.linalg.norm(grad) > 1e-5:

            beta = beta - 0.001 * grad
            for n in range(N):
        
                for c in range(C):
                    
                    conc1[n, c] = np.dot(np.dot(beta.T, O[c, n, :, :]), B[n, :])
                    conc2[n, c, :] = np.dot(O[c, n, :, :], B[n, :])

            conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1)[:, np.newaxis]
            grad = np.sum(- conc2[np.arange(N), Y.astype('int'), :] + \
                    np.sum(conc1[:, :, np.newaxis] * conc2, axis=1), axis=0) + \
                    2 * lambd * (beta - centroid)
            
            iter += 1

            if iter > 10:
                break

        return beta

    def multinomial_reg2(self, B_init, A, Y, O, lambd, centroid):
        
        M = A.shape[0]
        k = A.shape[1]
        C = O.shape[0]

        conc1 = np.zeros((M, C))
        conc2 = np.zeros((M, C, k))
        beta = B_init

        for m in range(M):

            for c in range(C):
                
                conc1[m, c] = np.dot(np.dot(A[m, :], O[c, :, :]), beta)
                conc2[m, c, :] = np.dot(O[c, :, :].T, A[m, :])

        conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1)[:, np.newaxis]
        
        grad = np.sum(- conc2[np.arange(M), Y.astype('int'), :] + \
                np.sum(conc1[:, :, np.newaxis] * conc2, axis=1), axis=0) + \
                2 * lambd * (beta - centroid)
        
        iter = 0
        while np.linalg.norm(grad) > 1e-5:

            beta = beta - 0.001 * grad
            
            for m in range(M):
        
                for c in range(C):
                    
                    conc1[m, c] = np.dot(np.dot(A[m, :], O[c, :, :]), beta)
                    conc2[m, c, :] = np.dot(O[c, :, :].T, A[m, :])

            conc1 = np.exp(conc1) / np.sum(np.exp(conc1), axis=1)[:, np.newaxis]
        
            grad = np.sum(- conc2[np.arange(M), Y.astype('int'), :] + \
                    np.sum(conc1[:, :, np.newaxis] * conc2, axis=1), axis=0) + \
                    2 * lambd * (beta - centroid)
            
            iter += 1
            if iter > 10:
                break

        return beta

    def cayley_transform(self, A, B, Y, O):

        G = np.zeros(O.shape)
        S = np.zeros(O.shape)
        C = G.shape[0]
        prob = np.zeros(Y.shape)

        for c in range(C):

            prob[:, c] = np.exp(np.sum(A * (B.dot(O[c, :, :].T)), axis=1))
        
        prob = prob / np.sum(prob, axis=1)[:, np.newaxis]
        coef = prob - Y

        iter = 0
        err = 1

        while err > 1e-2:
        
            for c in range(1, C): # the first set of rotation matrices are fixed as identity matrix for reference group

                G[c, :, :] = B.T.dot(np.diagflat(coef[:, c])).dot(A)
                S[c, :, :] = G[c, :, :].dot(O[c, :, :].T) - O[c, :, :].dot(G[c, :, :].T)
                O[c, :, :] = np.linalg.inv(np.eye(O.shape[1]) + 0.00005 * S[c, :, :]).dot(np.eye(O.shape[1]) - 0.00005 * S[c, :, :]).dot(O[c, :, :])

            for c in range(C):
        
                prob[:, c] = np.sum(np.exp(A.dot(O[c, :, :]) * B), axis=1)

            prob = prob / np.sum(prob, axis=1)[:, np.newaxis]
            coef = prob - Y

            err = np.max([np.linalg.norm(G[c, :, :]) for c in range(C)])
            iter += 1

            if iter > 10:
                break

        return O

    def _mc_fit(self, data, scheme="mv", maxiter=100, epsilon=1e-5, verbose=0):

        task, task_id = np.unique(data[:, 0], return_inverse=True)
        worker, worker_id = np.unique(data[:, 1], return_inverse=True)

        self._init_mc_params(data, scheme=scheme)

        iter = 0
        loss_cur = self._mc_loss_func(data)
        err = 1

        if verbose > 0:
            print("Iter: {0}, loss: {1}".format(iter, loss_cur))

        A_cur, B_cur, O_cur = self.A, self.B, self.O
        U_cur, V_cur = self.U, self.V

        while err > epsilon:

            A_prev, B_prev, O_prev = A_cur, B_cur, O_cur
            U_prev, V_prev = U_cur, V_cur
            Alpha, Beta = self._comp_centroid(A_prev, B_prev, U_prev, V_prev)

            loss_prev = loss_cur

            iter = iter + 1

            # update A
            
            for id, _ in enumerate(task):

                obsIdx = np.argwhere(task_id == id).squeeze()
                obsWorker = worker_id[obsIdx]
                obsB = B_prev[obsWorker, :]
                obsY = data[obsIdx, 2]
                obsO = O_prev[:, V_prev[obsWorker], :, :]
                if obsIdx.size < 2:
                    obsB = obsB[np.nexaxis, :]
                    obsY = np.array([obsY])

                A_cur[id, :] = self.multinomial_reg1(A_prev[id, :], obsB, obsY, obsO, self.lambda1, Alpha[id, :].T)
            
            self.A = A_cur
            
            # update B
            for jd, _ in enumerate(worker):

                obsIdx = np.argwhere(worker_id == jd).squeeze()
                obsTask = task_id[obsIdx]
                obsA = A_cur[obsTask, :]
                obsY = data[obsIdx, 2]
                obsO = O_prev[:, V_prev[jd], :, :]
                if obsIdx.size < 2:
                    obsA = obsA[np.newaxis, :]
                    obsY = np.array([obsY])

                B_cur[jd, :] = self.multinomial_reg2(B_prev[jd, :], obsA, obsY, obsO, self.lambda2, Beta[jd, :].T)
            self.B = B_cur
            
            # update O
            for jd in range(self.n_worker_group):
                
                obsIdx = np.argwhere(V_prev[worker_id] == jd).squeeze()
                obsTask = task_id[obsIdx]
                obsWorker = worker_id[obsIdx]
                obsA = A_cur[obsTask, :]
                obsB = B_cur[obsWorker, :]
                obsY = data[obsIdx, 2]
                obsY_dummy = np.zeros((len(obsY), self.n_task_group))
                obsY_dummy[np.arange(len(obsY)), obsY.astype('int')] = 1
                obsO = O_prev[:, jd, :, :]
                O_cur[:, jd, :, :] = self.cayley_transform(obsA, obsB, obsY_dummy, obsO)
            self.O = O_cur
            
            # update U
            U_cur = KMeans(n_clusters=self.n_task_group).fit_predict(A_cur)
            U_cur = self.label_swap(U_cur, U_prev)
            # update V
            V_cur = KMeans(n_clusters=self.n_worker_group).fit_predict(B_cur)
            V_cur = self.label_swap(V_cur, V_prev)

            self.A, self.B, self.O = A_cur, B_cur, O_cur
            self.U, self.V = U_cur, V_cur

            loss_cur = self._mc_loss_func(data)

            if verbose > 0:
                print("Iter: {0}, loss: {1}".format(iter, loss_cur))

            err = np.abs(loss_cur - loss_prev) / loss_prev

            if iter > maxiter:
                break

            self.iter = iter

        self.A, self.B, self.O, self.U, self.V = A_cur, B_cur, O_cur, U_cur, V_cur

    def fit(self, data, maxiter=100, epsilon=1e-3, verbose=0):

        self._prescreen(data)

        if self.n_task_group == 2:

            self._binary_fit(data, maxiter=maxiter, epsilon=epsilon, verbose=verbose)

            label = self._binary_infer(data)

            return label


        elif self.n_task_group > 2:

            self._mc_fit(data, maxiter=maxiter, epsilon=epsilon)

            label = self._mc_infer(data)

            return label


    def _binary_infer(self, data):

        gA, alpha = npi.group_by(self.U).mean(self.A, axis=0)
        gB, beta = npi.group_by(self.V).mean(self.B, axis=0)

        angle_matrix = alpha.dot(beta.T)
        idx = np.argmax(np.sum(np.abs(angle_matrix), axis=0))
        
        label = np.zeros((self.n_task, 2))
        label[:, 0] = np.unique(data[:, 0])
        label[self.U == gA[np.argmax(angle_matrix[:, idx])], 1] = 1

        high_quality_workers = np.argwhere(self.V == gB[idx]).squeeze()

        hq_index = np.zeros((self.n_worker, 2))
        hq_index[:, 0] = np.unique(data[:, 1])
        hq_index[:, 1] = (self.V == gB[idx]) * 1
 
        return label, high_quality_workers, hq_index

    def _mc_infer(self, data):

        # gA, alpha = npi.group_by(self.U).mean(self.A, axis=0)
        # gB, beta = npi.group_by(self.V).mean(self.B, axis=0)

        label = np.zeros((self.n_task, 2))
        label[:, 0] = np.unique(data[:, 0])

        # for i in gA:
        #     candidate = np.zeros((self.n_task_group, self.n_worker_group))
        #     for ii in range(self.n_task_group):
        #         for jj in range(self.n_worker_group):
        #             candidate[ii, jj] = alpha[ii, :].dot(self.O[i, jj, :, :]).dot(beta[jj, :])
            
        #     x = np.argwhere(candidate == np.max(candidate))

        label[:, 1] = self.U

        return label






