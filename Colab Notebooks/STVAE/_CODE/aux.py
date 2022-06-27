import torch
import numpy as np

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_float = map(float, strings.split(","))
    return tuple(mapped_float)

def process_args(parser):
    parser.add_argument('--fa', type=int, default=0, help='Type of weight feedback - 0 - bp, 1-fixed, 2-urfb')
    parser.add_argument('--use_multiple_gpus', type=int, default=None, help='Number of gpus to use')
    parser.add_argument('--image_levels', type=int, default=0, help='Image quantization levels')
    parser.add_argument('--new_dim', type=int, default=0, help='new image dimension')
    parser.add_argument('--future', type=int, default=0, help='how many other images to take into account in embedding loss')
    parser.add_argument('--patch_size', type=int, default=None, help='size of patch in gaze type augmentation')
    parser.add_argument('--clapp_dim',nargs="*", type=int, default=None, help='dimension of clapp matrix for SSL')
    parser.add_argument('--CC', type=float, default=1., help='inverse penalty in logistic regression')
    parser.add_argument('--transformation', default=None, help='type of transformation: aff or tps')
    parser.add_argument('--t_par', default='', help='suffix of temp file')
    parser.add_argument('--feats', type=int, default=0, help='Number of features in case data preprocessed')
    parser.add_argument('--crop', type=int, default=0, help='Size of random crop')
    parser.add_argument('--feats_back', action='store_true',help='reconstruct image from features')
    parser.add_argument('--filts', type=int, default=3, help='Filter size')
    parser.add_argument('--input_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--type', default='vae', help='type of transformation: aff or tps')
    parser.add_argument('--dataset', default='mnist', help='which data set')
    parser.add_argument('--hid_dataset', default=None, help='which data set')
    parser.add_argument('--hid_nepoch', type=int, default=100, help='which data set')
    parser.add_argument('--enc_layers',  nargs="*", default=None, help='encoder layer')
    parser.add_argument('--dec_layers_top',  nargs="*", default=None, help='decoder layer top')
    parser.add_argument('--dec_layers_bot',  nargs="*", default=None, help='decoder layer bottom')
    parser.add_argument('--dec_trans_top',  nargs="*", default=None, help='transform decoder layer')
    parser.add_argument('--layers',  nargs="*", default=None, help='main networ layers')
    parser.add_argument('--update_layers',nargs="*",default=None, help='layers to update')
    parser.add_argument('--copy_layers',nargs="*",default=None, help='layers to force copy')
    parser.add_argument('--no_copy_layers',nargs="*",default=None, help='layers to not copy')
    parser.add_argument('--hid_layers',  nargs="*", default=None, help='layer for network on embedding')
    parser.add_argument('--tps_num', type=int, default=3, help='dimension of s')
    parser.add_argument('--sdim', type=int, default=26, help='dimension of s')
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
    parser.add_argument('--optimizer_type', default='Adam', help='Type of optimiser')
    parser.add_argument('--lr', type=float, default=.001, help='Learning rate (default: .001)')
    parser.add_argument('--grad_clip', type=float, default=.0, help='clip gradient')
    parser.add_argument('--sparse', type=float, default=None, help='sparsity')
    parser.add_argument('--perturb', type=float, default=0, help='Learning rate (default: .001)')
    parser.add_argument('--hid_lr', type=float, default=.001, help='Learning rate (default: .001)')
    parser.add_argument('--binary_thresh', type=float, default=1e-6, help='threshold for bernoulli probs')
    parser.add_argument('--threshold', type=float, default=None, help='thresholding for images')
    parser.add_argument('--thr', type=float, default=4., help='threshold for contrastive learning with hinge loss')
    parser.add_argument('--delta', type=float, default=2., help='margin for contrastive learning with hinge loss')
    parser.add_argument('--alpha', type=float, default=.9, help='history weight for covariance estimate in direct ssl')
    parser.add_argument('--eps', type=float, default=.5, help='identity weight for covariance estimate in direct ssl')
    parser.add_argument('--conf', type=float, default=0, help='confidence level')
    parser.add_argument('--ortho_lr', type=float, default=.0, help='Learning rate (default: .000001)')
    parser.add_argument('--mu_lr', type=float, default=[.05,.01], nargs=2,help='Learning rate (default: .05)')
    parser.add_argument('--s_factor', type=float, default=4.0, help='weight decay')
    parser.add_argument('--h_factor', type=float, default=.2, help='weight decay')
    parser.add_argument('--lamda', type=float, default=1.0, help='weight decay')
    parser.add_argument('--lamda1', type=float, default=1.0, help='penalty on conv matrix')
    parser.add_argument('--scale', type=float, default=None, help='range of bias term for decoder templates')
    parser.add_argument('--penalize_activations', type=float,default=None, help='penalize activations of layers')
    parser.add_argument('--lim', type=int, default=0, help='penalty on conv matrix')
    parser.add_argument('--num_mu_iter', type=int, default=10, help='Learning rate (default: .05)')
    parser.add_argument('--wd', type=float, default=0, help='Use weight decay')
    parser.add_argument('--sched', type=tuple_type, default="(0.,0.)",  help='time step change')
    parser.add_argument('--hid_sched', type=float, default=[0.,0.], nargs=2, help='time step change')
    parser.add_argument('--cl', type=int, default=None, help='class (default: None)')
    parser.add_argument('--run_existing', action='store_true', help='Use existing model')
    parser.add_argument('--nti', type=int, default=500, help='num test iterations (default: 100)')
    parser.add_argument('--nvi', type=int, default=20, help='num val iterations (default: 20)')
    parser.add_argument('--n_mix', type=int, default=0, help='num mixtures (default: 0)')
    parser.add_argument('--hidden_clusters', type=int, default=10, help='num mixtures (default: 0)')
    parser.add_argument('--clust', type=int, default=None, help='which cluster to shoe')
    parser.add_argument('--OPT', action='store_true', help='Optimization instead of encoding')
    parser.add_argument('--opt_jump', type=int, default=1, help='dimension of part')
    parser.add_argument('--network', action='store_true', help='classification network')
    parser.add_argument('--CONS', action='store_true', help='Output to consol')
    parser.add_argument('--layerwise', action='store_true', help='Do layerwise processing')
    parser.add_argument('--randomize_layers', nargs="*",default=None, help='layers to choose randomly')
    parser.add_argument('--hinge', action='store_true', help='Output to consol')
    parser.add_argument('--use_clutter_model', type=int, default=None, help='use clutter model')
    parser.add_argument('--sample', action='store_true', help='sample from distribution')
    parser.add_argument('--cluster_hidden', action='store_true', help='cluster latent variables')
    parser.add_argument('--classify', action='store_true', help='Output to consol')
    parser.add_argument('--Diag', action='store_true', help='The linear layer is diagonal')
    parser.add_argument('--Iden', action='store_true', help='The linear layer is initizlized as identity')
    parser.add_argument('--output_cont', type=float, default=0., help='factor in front of MSE loss')
    parser.add_argument('--erode', type=int, default=[0,0,0], nargs=3, help='add stuff to image')
    parser.add_argument('--cont_training', action='store_true', help='continue training')
    parser.add_argument('--del_last', action='store_true', help='dont update classifier weights')
    parser.add_argument('--nosep', action='store_true', help='separate optimization in VAE OPT')
    parser.add_argument('--embedd', action='store_true', help='embedding training')
    parser.add_argument('--embedd_type', default='new', help='embedding cost type')
    parser.add_argument('--embedd_layer', default=None, help='embedding layer')
    parser.add_argument('--reinit', action='store_true', help='reinitialize part of trained model')
    parser.add_argument('--only_pi', action='store_true', help='only optimize over pi')
    parser.add_argument('--block', action='store_true', help='block one branch of embedding comp in SSL')
    parser.add_argument('--double_aug', action='store_true', help='augment both branches in SSL')
    parser.add_argument('--verbose', action='store_true', help='only optimize over pi')
    parser.add_argument('--edges', action='store_true', help='compute edges')
    parser.add_argument('--no_standardize', action='store_true', help='standardize embeddings')
    parser.add_argument('--AVG', type=int, default=None, help='average pool2d output of embeddings')
    parser.add_argument('--deform', action='store_true', help='show deformations')
    parser.add_argument('--lower_decoder', action='store_true', help='start decoding from lowerlayer')
    parser.add_argument('--randomize', action='store_true', help='start decoding from lowerlayer')
    parser.add_argument('--L1', action='store_true', help='use L1 loss')
    parser.add_argument('--hid_model', action='store_true', help='classifier on top of embedding')
    parser.add_argument('--bn', default='none', help='type of batch normalizations')
    parser.add_argument('--edge_dtr', type=float, default=0., help='difference minimum for edge')
    parser.add_argument('--out_file', default='OUT.txt',help='output stuff')
    parser.add_argument('--output_prefix', default='', help='path to model')
    parser.add_argument('--show_weights',default=None,help='which weights to show')

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



