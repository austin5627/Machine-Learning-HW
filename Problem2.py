import numpy as np
import cvxopt


cvxopt.solvers.options["abstol"] = 1e-9
cvxopt.solvers.options["reltol"] = 1e-9
cvxopt.solvers.options["feastol"] = 1e-9


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))


def featurize_data(data):
    featured = np.ones((data.shape[0], 10))
    for i, e in enumerate(data):
        featured[i] = [1, e[0], e[1], e[2], e[3], e[0]**2, e[1]**2, e[2]**2, e[3]**2, e[4]]
    return featured


def find_support_vectors(data):
    num_vectors = 0
    vectors = []
    for i, m in enumerate(data):
        xm = m[0:9]
        y = m[9]
        if y * (np.dot(w, xm) + b) - 1 <= 2e-13:
            vectors.append(i)
    return vectors


data = np.loadtxt("mystery.data", delimiter=",")
print(data.shape)
featureized_data = featurize_data(data)
print(featureized_data)

P = np.identity(10)
P[9, 9] = 0
q = np.zeros([featureized_data.shape[1]])

G = np.array([np.append(-x[0:9]*x[9], -x[9]) for x in featureized_data], dtype=float)
h = np.array([-1.0] * featureized_data.shape[0])

out = cvxopt_solve_qp(P, q, G, h)
w = out[0:9]
b = out[9]

print(f'\n\nw is \n{w}')
print(f'b is {b}')

s_v = find_support_vectors(featureized_data)
for i in s_v:
    print('\n\n')
    print(f'Support vector {i} is \n{data[i, 0:4]}\nwith value of {data[i, 4]}')
    print(f'Featurized Support vector is \n{featureized_data[i, 0:8]}')
margin = 1/np.linalg.norm(w)
print(f'\n\nThe optimal margin is {margin}')

