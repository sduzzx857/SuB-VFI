from dataloader.particle import Particle_Flow

def build_train_dataset(args):

    if args.stage == 'particle':
        train_dataset = Particle_Flow(root=args.root, dataset_name=args.dataset, snr=args.snr, density=args.density, npoints=args.npoints)
    else:
       raise ValueError(f'stage {args.stage} is not supported')

    return train_dataset
