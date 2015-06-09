#!/usr/bin/env python

'''
    author: Kyle Wendt <kylewendt@gmail.com>
    copyright: 2015 Kyle Wendt
    License: GPL v3
'''


def main():
    import argparse, textwrap
    from numpy import arange, array, sqrt

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
     epilog=textwrap.dedent('''\
        Example:
        For odd parity Lithium-7 states, Ltot = 3, N = 4, and Z = 3, with Nmax upto 40
        python ncsm_ho_ir_scale.py -N 40 3 4 3'''))
    parser.add_argument('Ltot', metavar='Ltot', type=int, help='Sum of orbital angular momenta for "lowest" HO state, Partity will be determined partiy of Ltot (Pi=-1^Ltot)')
    parser.add_argument('N', metavar='N', type=int, help='number of neutrons')
    parser.add_argument('Z', metavar='Z', type=int, help='number of protons')
    parser.add_argument('-N', '--ntot', type=int, dest='Ntot',  help='Largest Ntot to output (default 40)', default=40)
    args = parser.parse_args()

    N, Z, l_total = args.N, args.Z, args.Ltot
    par = 1 if l_total & 1 else 0
    Pi = -1 if l_total & 1 else 1
    A = N + Z
    n_tot = min(40, abs(args.Ntot))

    d_tot = 3 * A
    d_int = 3 * (A - 1)
    two_l_tot = 2 * l_total + (d_tot - 3)
    two_l_int = 2 * l_total + (d_int - 3)

    k2_all = compute_k2_vals((two_l_tot + 1) // 2, 1)

    k2_tot = k2_all[two_l_tot]
    k2_int = k2_all[two_l_int]

    n_tot_min = l_total + ((l_total ^ par) & 1)
    n_all = arange(l_total + ((l_total ^ par) & 1), n_tot + 1, 2)
    ho_k2_tot = array([HH_HO_eigvals(n, l_total, d_tot)[0] for n in n_all])
    ho_k2_int = array([HH_HO_eigvals(n, l_total, d_int)[0] for n in n_all])

    nt_tot = sqrt(k2_tot / ho_k2_tot)
    nt_int = sqrt(k2_int / ho_k2_int)

    print_header(N, Z, Pi, l_total)
    for N, Kt, Ki in zip(n_all, nt_tot, nt_int):
        print r"  {:4}  {:16.8f}  {:16.8f}".format(N, Kt, Ki)

    # K2 = Kzeros[2 * Ltot + 3 * A]
def print_header(n, z, pi, l):
    print r"# L_{\rm eff} = b * \tilde{N}"
    print r"# \Lambda_{\rm eff} = b^{-1} * \tilde{N}"
    print r"# N: {:d}  Z: {:d}  Pi: {:+d}  L total: {:d}".format(n, z, pi, l)
    print "# {:>4s}  {:>16s}  {:>16s}".format("Ntot", r"\tilde{N}", r"\tilde{N}_{int}")


def compute_k2_vals(l_max, num_vals):
    """
    Compute hyper radial infinite well K^2 eigenvalues for a well of unit radial width.  The eigenvalues for a well with
     parameter L = G + 3 D / 2

    Compute square of zeros of J_{l+1/2}(x) for l = 0, 1/2, 1, ..., floor(l_max), floor(l_max)+1/2
    :param l_max:  Max l to find zeros of
    :param num_vals:  Total number of zeros to find for each l
    :return K2:  A 2*l_max + 1 by num_vals ndarray containing the computed squared zeros.  K2[2*K + D-3] are the
    eigenvalues for dimension D and hyper angular momentum L
    """

    from numpy import arange, pi, zeros, zeros_like
    from scipy.optimize import brentq
    from scipy.special import jv

    zro = zeros((2 * l_max + 1, num_vals), dtype=float)

    z_l_m_1 = pi * arange(1,num_vals + l_max + 1)
    z_l = zeros_like(z_l_m_1)
    zz_l = zeros_like(z_l_m_1)

    zro[0] = z_l_m_1[:num_vals]

    for l in range(1, l_max + 1):
        for i in range(num_vals + l_max - l):
            zz_l[i] = brentq(lambda x: jv(l, x), z_l_m_1[i], z_l_m_1[i + 1])
            z_l[i] = brentq(lambda x: jv(l + .5, x), z_l_m_1[i], z_l_m_1[i + 1])

        z_l_m_1[:] = z_l[:]
        zro[2 * l] = z_l[:num_vals]
        zro[2 * l - 1] = zz_l[:num_vals]
    if num_vals == 1:
        zro = zro[:,0]
    return zro**2


def HH_HO_eigvals(NMax, K, D):
    from numpy import arange, sqrt, vstack
    from scipy.linalg import eigvals_banded
    nmax = (NMax-K) // 2
    n = arange(nmax+1)
    return eigvals_banded(vstack((2 * n + K + D / 2., sqrt((n + 1) * (n + K + D / 2.)))), lower=True)

if __name__ == '__main__':
    main()
