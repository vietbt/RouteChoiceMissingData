import argparse
from trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument('--train_mu', action='store_true')
    parser.add_argument('--use_LS', action='store_true')
    parser.add_argument('--use_LS_for_beta', action='store_true')
    parser.add_argument('--prob', type=float)
    parser.add_argument('--method', type=str)
    parser.add_argument('--logdir', type=str)

    args = parser.parse_args()

    init_beta = [-2.494, -0.933, -0.411, -4.559]
    init_scale = [0.0, 0.0, 0.0]

    trainer = Trainer(init_beta, init_scale, args.prob, args.method, args.use_LS, args.use_LS_for_beta, args.seed, train_mu=args.train_mu, device='cuda', logdir=args.logdir)
    if args.prob >= 0.5 or args.use_LS_for_beta:
        trainer.train(use_missing=False, use_LS=False, use_LS_for_beta=args.prob<0.7, optim='lbfgs')
    trainer.train(use_missing=False, use_LS_for_beta=True, optim='lbfgs2')
    trainer.train(use_missing=False, use_LS_for_beta=True, optim='lbfgs3')
    trainer.train(use_missing=True, optim='lbfgs4')

