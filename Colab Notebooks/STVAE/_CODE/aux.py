import torch
import numpy as np

def process_args(parser):
    parser.add_argument('--fa', type=int, default=0, help='Type of weight feedback - 0 - bp, 1-fixed, 2-urfb')
    parser.add_argument('--image_levels', type=int, default=0, help='Image quantization levels')
    parser.add_argument('--full_dim', type=int, default=256, help='fully connected layer size')
    parser.add_argument('--hid_hid', type=int, default=256, help='fully connected layer size')
    parser.add_argument('--hid_drop', type=float, default=0., help='dropout')
    parser.add_argument('--transformation', default='aff', help='type of transformation: aff or tps')
    parser.add_argument('--feats', type=int, default=0, help='Number of features in case data preprocessed')
    parser.add_argument('--feats_back', action='store_true',help='reconstruct image from features')
    parser.add_argument('--filts', type=int, default=3, help='Filter size')
    parser.add_argument('--pool', type=int, default=2, help='Pooling size')
    parser.add_argument('--pool_stride', type=int, default=2, help='Pooling stride')
    parser.add_argument('--input_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--type', default='vae', help='type of transformation: aff or tps')
    parser.add_argument('--dataset', default='mnist', help='which data set')
    parser.add_argument('--hid_dataset', default=None, help='which data set')
    parser.add_argument('--hid_nepoch', type=int, default=100, help='which data set')
    parser.add_argument('--enc_layers',  nargs="*", default=None, help='encoder layer')
    parser.add_argument('--layers',  nargs="*", default=None, help='main networ layers')
    parser.add_argument('--update_layers',nargs="*",default=None, help='layers to update')
    parser.add_argument('--hid_layers',  nargs="*", default=None, help='layer for network on embedding')
    parser.add_argument('--tps_num', type=int, default=3, help='dimension of s')
    parser.add_argument('--sdim', type=int, default=26, help='dimension of s')
    parser.add_argument('--hdim', type=int, default=256, help='dimension of h')
    parser.add_argument('--nepoch', type=int, default=40, help='number of training epochs')
    parser.add_argument('--gpu', type=int, default=1, help='whether to run in the GPU')
    parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')
    parser.add_argument('--num_train', type=int, default=60000, help='num train (default: 60000)')
    parser.add_argument('--num_sample', type=int, default=10000, help='num samples')
    parser.add_argument('--network_num_train', type=int, default=60000, help='num train (default: 60000)')
    parser.add_argument('--num_test', type=int, default=0, help='num test (default: 10000)')
    parser.add_argument('--nval', type=int, default=0, help='num train (default: 1000)')
    parser.add_argument('--mb_size', type=int, default=100, help='mb_size (default: 500)')
    parser.add_argument('--n_class', type=int, default=0, help='number of classes')
    parser.add_argument('--par_file', default='t_par', help='default parameter file')
    parser.add_argument('--model', default=None, nargs="*", help='model (default: base)')
    parser.add_argument('--model_out', default=None, help='model (default: base)')
    parser.add_argument('--optimizer', default='Adam', help='Type of optimiser')
    parser.add_argument('--lr', type=float, default=.001, help='Learning rate (default: .001)')
    parser.add_argument('--grad_clip', type=float, default=.0, help='clip gradient')
    parser.add_argument('--perturb', type=float, default=0, help='Learning rate (default: .001)')
    parser.add_argument('--hid_lr', type=float, default=.001, help='Learning rate (default: .001)')
    parser.add_argument('--binary_thresh', type=float, default=1e-6, help='threshold for bernoulli probs')
    parser.add_argument('--conf', type=float, default=0, help='confidence level')
    parser.add_argument('--ortho_lr', type=float, default=.0, help='Learning rate (default: .000001)')
    parser.add_argument('--mu_lr', type=float, default=[.05,.01], nargs=2,help='Learning rate (default: .05)')
    parser.add_argument('--s_factor', type=float, default=4.0, help='weight decay')
    parser.add_argument('--h_factor', type=float, default=.2, help='weight decay')
    parser.add_argument('--lamda', type=float, default=1.0, help='weight decay')
    parser.add_argument('--lamda1', type=float, default=1.0, help='penalty on conv matrix')
    parser.add_argument('--scale', type=float, default=None, help='range of bias term for decoder templates')
    parser.add_argument('--h2pi_scale', type=float, default=1., help='scaling of h2pi initial matrix')
    parser.add_argument('--penalize_activations', type=float,default=None, help='penalize activations of layers')
    parser.add_argument('--lim', type=int, default=0, help='penalty on conv matrix')
    parser.add_argument('--num_mu_iter', type=int, default=10, help='Learning rate (default: .05)')
    parser.add_argument('--wd', type=float, default=0, help='Use weight decay')
    parser.add_argument('--sched', type=float, default=[0.,0.], nargs=2, help='time step change')
    parser.add_argument('--cl', type=int, default=None, help='class (default: None)')
    parser.add_argument('--run_existing', action='store_true', help='Use existing model')
    parser.add_argument('--nti', type=int, default=500, help='num test iterations (default: 100)')
    parser.add_argument('--nvi', type=int, default=20, help='num val iterations (default: 20)')
    parser.add_argument('--n_mix', type=int, default=0, help='num mixtures (default: 0)')
    parser.add_argument('--hidden_clusters', type=int, default=10, help='num mixtures (default: 0)')
    parser.add_argument('--clust', type=int, default=None, help='which cluster to shoe')
    parser.add_argument('--n_parts', type=int, default=0, help='number of parts per location')
    parser.add_argument('--n_part_locs', type=int, default=0, help='number of part locations (a^2)')
    parser.add_argument('--part_dim', type=int, default=None, help='dimension of part')
    parser.add_argument('--MM', action='store_true', help='Use max max')
    parser.add_argument('--OPT', action='store_true', help='Optimization instead of encoding')
    parser.add_argument('--opt_jump', type=int, default=1, help='dimension of part')
    parser.add_argument('--network', action='store_true', help='classification network')
    parser.add_argument('--CONS', action='store_true', help='Output to consol')
    parser.add_argument('--decoder_gaus', default=None, help='extra decoder layer to get correlated gaussians')
    parser.add_argument('--layerwise', action='store_true', help='Do layerwise processing')
    parser.add_argument('--layerwise_randomize', nargs="*",default=None, help='layers to choose randomly')
    parser.add_argument('--hinge', action='store_true', help='Output to consol')
    parser.add_argument('--sample', action='store_true', help='sample from distribution')
    parser.add_argument('--cluster_hidden', action='store_true', help='cluster latent variables')
    parser.add_argument('--classify', action='store_true', help='Output to consol')
    parser.add_argument('--Diag', action='store_true', help='The linear layer is diagonal')
    parser.add_argument('--Iden', action='store_true', help='The linear layer is initizlized as identity')
    parser.add_argument('--output_cont', type=float, default=0., help='factor in front of MSE loss')
    parser.add_argument('--erode', action='store_true', help='cont data')
    parser.add_argument('--cont_training', action='store_true', help='continue training')
    parser.add_argument('--del_last', action='store_true', help='dont update classifier weights')
    parser.add_argument('--nosep', action='store_true', help='separate optimization in VAE OPT')
    parser.add_argument('--embedd', action='store_true', help='embedding training')
    parser.add_argument('--embedd_type', default='new', help='embedding cost type')
    parser.add_argument('--embedd_layer', default=None, help='embedding layer')
    parser.add_argument('--reinit', action='store_true', help='reinitialize part of trained model')
    parser.add_argument('--only_pi', action='store_true', help='only optimize over pi')
    parser.add_argument('--gauss_prior', action='store_true', help='only optimize over pi')
    parser.add_argument('--edges', action='store_true', help='compute edges')
    parser.add_argument('--bn', default='none', help='type of batch normalizations')
    parser.add_argument('--edge_dtr', type=float, default=0., help='difference minimum for edge')
    parser.add_argument('--out_file', default='OUT.txt',help='output stuff')
    parser.add_argument('--output_prefix', default='', help='path to model')
    #parser.add_argument('--file', type=open, default='/ME/My Drive/Colab Notebooks/STVAE/_pars/pars_mnist', action=LoadFromFile)
    #parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    #args = parser.parse_args()

    return (parser)



    def get_binary_signature(model,inp1, inp2=None, lays=[]):

        num_tr1=inp1[0].shape[0]
        OT1=[];
        with torch.no_grad():
            for j in np.arange(0, num_tr1, model.bsz, dtype=np.int32):
                data=(torch.from_numpy(inp1[0][j:j + model.bsz]).float()).to(model.dv)
                out,ot1=model.forward(data,everything=True)
                OTt=[]
                for l in lays:
                    OTt+=[ot1[l].reshape(model.bsz,-1)]
                OT1+=[torch.cat(OTt,dim=1)]

            OT1 = torch.cat(OT1)
            qq1=2*(OT1.reshape(num_tr1,-1)>0).type(torch.float32)-1.

            if inp2 is not None:
                OT2 = []
                num_tr2=inp2[0].shape[0]
                for j in np.arange(0, num_tr2, model.bsz, dtype=np.int32):
                    data = (torch.from_numpy(inp2[0][j:j + model.bsz]).float()).to(model.dv)
                    out, ot2 = model.forward(data, everything=True)
                    OTt = []
                    for l in lays:
                        OTt += [ot2[l].reshape(model.bsz, -1)]
                    OT2 += [torch.cat(OTt,dim=1)]
                OT2=torch.cat(OT2)
                qq2=2*(OT2.reshape(num_tr2,-1)>0).type(torch.float32)-1.
            else:
                qq2=qq1

            cc=torch.mm(qq1,qq2.transpose(0,1))/qq1.shape[1]
            if inp2 is None:
                cc-=torch.diag(torch.diag(cc))
            ##print('CC',torch.sum(cc==1.).type(torch.float32)/num_tr1)
            return cc



