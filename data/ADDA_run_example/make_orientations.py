import numpy as np
import argparse


def main(args):
    if args.orientations <= 0:
        p.error("--orientations must be greater than zero")
            
    rng = np.random.default_rng(args.seed)
    alphas = rng.uniform(0.0, 360.0, args.orientations)
    cos_betas = rng.uniform(-1.0, 1.0, args.orientations)
    betas = np.degrees(np.arccos(cos_betas))
    gammas = rng.uniform(0.0, 360.0, args.orientations)

    np.savetxt(args.alphas_output, alphas, fmt="%.0f")
    np.savetxt(args.betas_output, betas, fmt="%.0f")
    np.savetxt(args.gammas_output, gammas, fmt="%.0f")


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--orientations", type=int, required=True)
    p.add_argument("--alphas_output", type=str, default="alphas.txt")
    p.add_argument("--betas_output", type=str, default="betas.txt")
    p.add_argument("--gammas_output", type=str, default="gammas.txt")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    main(args)
