from losses import *
from torch import nn, optim
from network import network
from get_net_text import get_network

class temp_args(nn.Module):
    def __init__(self):
        super(temp_args, self).__init__()
        self.back = None
        self.first = 0
        self.everything = False
        self.layer_text = None
        self.dv = None
        self.optimizer = None
        self.embedd_layer = None
        KEYS = None


def initialize_model(args, sh, layers, device, layers_dict=None):
    model = network()
    # if not read_model:
    if layers_dict == None:
        layers_dict = get_network(layers)
    # else:
    #     print('LOADING OLD MODEL')
    #     sm = torch.load(datadirs + '_output/' + args.model + '.pt', map_location='cpu')
    #     args=sm['args']
    #     # model_old = network.network()
    for l in layers_dict:
        if 'dense_gaus' in l['name']:
            if sh is not None:
                l['num_units'] = sh[0]

    atemp = temp_args()
    atemp.layer_text = layers_dict
    atemp.dv = device
    atemp.everything = False
    atemp.bn = args.bn
    atemp.fout = args.fout
    atemp.slope = args.slope
    atemp.fa = args.fa
    atemp.embedd_type = args.embedd_type
    atemp.randomize_layers = args.randomize_layers
    atemp.penalize_activations = args.penalize_activations
    if args.hinge:
        atemp.loss = hinge_loss(num_class=args.num_class)
    else:
        atemp.loss = nn.CrossEntropyLoss()

    if args.crop and len(sh) == 3:
        sh = (sh[0], args.crop, args.crop)
        print(sh)
    if args.embedd_type == 'clapp':
        if args.clapp_dim is not None:
            model.add_module('clapp', nn.Conv2d(args.clapp_dim[1], args.clapp_dim[1], 1))
        if args.update_layers is not None:
            args.update_layers.append('clapp')
    if args.embedd_type == 'AE' or args.embedd_type is not None:
        atemp.everything = True

    if sh is not None:
        temp = torch.zeros([1] + list(sh))  # .to(device)
        # Run the network once on dummy data to get the correct dimensions.
        atemp.first = 1
        atemp.input_shape = None
        bb = model.forward(temp, atemp)
        if args.embedd_type == 'clapp':
            args.clapp_dim = atemp.clapp_dim

        atemp.output_shape = bb[0].shape
        if atemp.input_shape is None:
            atemp.input_shape = sh

        if 'ae' not in args.type:
            # print(self.layers, file=self.fout)
            tot_pars = 0
            KEYS = []
            for keys, vals in model.named_parameters():
                if 'running' not in keys and 'tracked' not in keys:
                    KEYS += [keys]
                # tot_pars += np.prod(np.array(vals.shape))

            # TEMPORARY
            if True:
                pp = []
                atemp.KEYS = KEYS
                for k, p in zip(KEYS, model.parameters()):
                    if (args.update_layers is None):
                        if atemp.first == 1:
                            atemp.fout.write('TO optimizer ' + k + str(np.array(p.shape)) + '\n')
                        tot_pars += np.prod(np.array(p.shape))
                        pp += [p]
                    else:
                        found = False
                        for u in args.update_layers:
                            if u == k.split('.')[1] or u == k.split('.')[0]:
                                found = True
                                if atemp.first == 1:
                                    atemp.fout.write('TO optimizer ' + k + str(np.array(p.shape)) + '\n')
                                tot_pars += np.prod(np.array(p.shape))
                                pp += [p]
                        if not found:
                            p.requires_grad = False
                if atemp.first == 1:
                    atemp.fout.write('tot_pars,' + str(tot_pars) + '\n')
                if 'ae' not in args.type:
                    if (args.optimizer_type == 'Adam'):
                        if atemp.first == 1:
                            atemp.fout.write('Optimizer Adam ' + str(args.lr) + '\n')
                        atemp.optimizer = optim.Adam(pp, lr=args.lr, weight_decay=args.wd)
                    else:
                        if atemp.first == 1:
                            atemp.fout.write('Optimizer SGD ' + str(args.lr))
                        atemp.optimizer = optim.SGD(pp, lr=args.lr, weight_decay=args.wd)

        atemp.first = 0
        bsz = args.mb_size
        if args.embedd_type is not None:
            if args.embedd_type == 'L1dist_hinge':
                atemp.loss = L1_loss(atemp.dv, bsz, args.future, args.thr, args.delta, WW=1., nostd=True)
            elif args.embedd_type == 'clapp':
                atemp.loss = clapp_loss(atemp.dv)
            elif args.embedd_type == 'binary':
                atemp.loss = binary_loss(atemp.dv)
            elif args.embedd_type == 'direct':
                atemp.loss = direct_loss(bsz, atemp.output_shape[1], alpha=args.alpha, eps=args.eps, device=atemp.dv)
            elif args.embedd_type == 'barlow':
                atemp.loss = barlow_loss(bsz, standardize=not args.no_standardize, device=atemp.dv, lamda=args.lamda)
            elif args.embedd_type == 'orig':
                atemp.loss = simclr_loss(atemp.dv, bsz)
            elif args.embedd_type == 'AE':
                atemp.loss = AE_loss(lamda=args.lamda, l1=args.L1)
        model.add_module('temp', atemp)
        if args.use_multiple_gpus is not None:
            bsz = bsz // args.use_multiple_gpus
        model = model.to(atemp.dv)
        return model






