
# Symbolic FLOP estimates for each architecture
# Based on Table 4.1 from "The Cost-Accuracy Trade-Off In Operator Learning With Neural Networks"

def pca_net_flops(Np, du, dv, w):
    return du * (2 * Np - 1) + 2 * du * w + 4 * w**2 + 2 * dv * w + 3 * w + (2 * dv - 1) * Np

def deep_onet_flops(Np, du, dv, w):
    return du * (2 * Np - 1) + 2 * du * w + 4 * w**2 + 2 * dv * w + 3 * w + (2 * dv - 1) * Np

def para_net_flops(Np, du, dy, do, w):
    return du * (2 * Np - 1) + (2 * (du + dy) * w + 4 * w**2 + 2 * w * do + 3 * w) * Np

def fno_flops(Np, di, do, df, kmax):
    return (
        2 * Np * df * (di + do) +
        3 * (10 * df * Np * (Np.bit_length() - 1) + kmax * (2 * df**2 - df) + 2 * df**2 * Np)
    )

