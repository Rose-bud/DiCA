import numpy as np
import os
import scipy.io as sio

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch

from utils.config import args
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
cudnn.benchmark = True
import nets as models
from utils.bar_show import progress_bar
from src.partialdataset import cross_modal_dataset
import src.utils as utils
import scipy
import scipy.spatial
import torch.nn.functional as F
from utils import *
import random


best_acc = 0
best_epoch = -1

args.log_dir = os.path.join(args.root_dir, 'logs', args.log_name)
args.ckpt_dir = os.path.join(args.root_dir, 'ckpt', args.ckpt_dir)

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)

def load_dict(model, path):
    chp = torch.load(path)
    state_dict = model.state_dict()
    for key in state_dict:
        if key in chp['model_state_dict']:
            state_dict[key] = chp['model_state_dict'][key]
    model.load_state_dict(state_dict)

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    print('===> Preparing data ..')
    train_dataset = cross_modal_dataset(args.data_name, args.partial_ratio, 'train', partial_file = os.path.join(args.root_dir, 'data', args.partial_file))

    SEED = args.seed
    seed_torch(SEED)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,

        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )

    valid_dataset = cross_modal_dataset(args.data_name, args.partial_ratio, 'valid')
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=False
    )

    test_dataset = cross_modal_dataset(args.data_name, args.partial_ratio, 'test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=False
    )

    print('===> Building Models..')
    multi_models = []
    n_view = len(train_dataset.train_data)
    for v in range(n_view):
        if v == args.views.index('Img'): # Images
            multi_models.append(models.__dict__['ImageNet'](input_dim=train_dataset.train_data[v].shape[1], output_dim=args.output_dim).cuda())
        elif v == args.views.index('Txt'): # Text
            multi_models.append(models.__dict__['TextNet'](input_dim=train_dataset.train_data[v].shape[1], output_dim=args.output_dim).cuda())
        else:
            multi_models.append(models.__dict__['ImageNet'](input_dim=train_dataset.train_data[v].shape[1], output_dim=args.output_dim).cuda())

    C = torch.Tensor(args.output_dim, args.output_dim)
    C = torch.nn.init.orthogonal(C, gain=1)[:, 0: train_dataset.class_num].cuda()
    C.requires_grad = True

    embedding = torch.eye(train_dataset.class_num).cuda()
    embedding.requires_grad = False

    parameters = [C]
    for v in range(n_view):
        parameters += list(multi_models[v].parameters())
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(parameters, lr=args.lr, betas=[0.5, 0.999], weight_decay=args.wd)

    lr_schedu = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=0, last_epoch=-1)

    summary_writer = SummaryWriter(args.log_dir)

    if args.resume:
        ckpt = torch.load(os.path.join(args.ckpt_dir, args.resume))
        for v in range(n_view):
            multi_models[v].load_state_dict(ckpt['model_state_dict_%d' % v])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print('===> Load last checkpoint data')

    else:
        start_epoch = 0
        print('===> Start from scratch')

    def set_train():
        for v in range(n_view):
            multi_models[v].train()

    def set_eval():
        for v in range(n_view):
            multi_models[v].eval()

    def train(epoch, confidence_img, confidence_txt, all_partial_targets, all_true_targets, prototypes_img, prototypes_txt):

        print('\nEpoch: %d / %d' % (epoch, args.max_epochs))
        set_train()
        train_loss, loss_list, correct_list, total_list = 0., [0.] * n_view, [0.] * n_view, [0.] * n_view

        for batch_idx, (batches, partial_targets, true_targets, index) in enumerate(train_loader):

            batches, partial_targets, true_targets = [batches[v].cuda() for v in range(n_view)], [partial_targets[v].cuda() for v in range(n_view)], [true_targets[v].cuda() for v in range(n_view)]
            norm = C.norm(dim=0, keepdim=True)
            C.data = (C / norm).detach()

            for v in range(n_view):
                multi_models[v].zero_grad()
            optimizer.zero_grad()

            logits0, outputs0 = [multi_models[v](batches[v]) for v in range(n_view)]
            logits = [logits0[0], outputs0[0]]
            outputs = [logits0[1], outputs0[1]]

            if args.method == "ours":

                pred_scores_img = torch.softmax(outputs[0], dim=1) * partial_targets[0]
                pred_scores_img_norm = pred_scores_img / pred_scores_img.sum(dim=1).repeat(args.num_class, 1).transpose(0, 1)
                _, pseudo_labels_img = torch.max(pred_scores_img_norm, dim=1)

                pred_scores_txt = torch.softmax(outputs[1], dim=1) * partial_targets[1]
                pred_scores_txt_norm = pred_scores_txt / pred_scores_txt.sum(dim=1).repeat(args.num_class, 1).transpose(0, 1)
                _, pseudo_labels_txt = torch.max(pred_scores_txt_norm, dim=1)


                prototypes_img = prototypes_img.detach()
                prototypes_txt = prototypes_txt.detach()

                proto_weight = utils.set_prototype_update_weight(epoch, args)

                for feat_i, label_i in zip(outputs[0], pseudo_labels_img):
                    prototypes_img[label_i] = proto_weight * prototypes_img[label_i] + (1 - proto_weight) * feat_i
                for feat_t, label_t in zip(outputs[1], pseudo_labels_txt):
                    prototypes_txt[label_t] = proto_weight * prototypes_txt[label_t] + (1 - proto_weight) * feat_t

                prototypes_img = F.normalize(prototypes_img, p=2, dim=1)
                prototypes_txt = F.normalize(prototypes_txt, p=2, dim=1)

                loss_icc = utils.supervised_contrastive_loss(outputs, torch.stack((pred_scores_img_norm, pred_scores_txt_norm),dim=0),tau=args.tau)

                loss_cls_img = utils.nbd_loss( outputs[0] , all_partial_targets[0][index], pred_scores_img_norm)
                loss_cls_txt = utils.nbd_loss( outputs[1] , all_partial_targets[1][index], pred_scores_txt_norm)
                loss_cls = loss_cls_img + loss_cls_txt

                loss_pca_src = utils.compute_pda_loss(outputs[0], prototypes_img, prototypes_txt)
                loss_pca_tar = utils.compute_pda_loss(outputs[1], prototypes_img, prototypes_txt)
                loss_pca = loss_pca_src + loss_pca_tar

            loss = 0
            if args.method =="ours":
                loss =  args.w1 * loss_cls + args.w2 * loss_icc + args.w3 * loss_pca

            if args.method == "ours":
                confidence_img = utils.confidence_update(confidence_img, outputs[0].clone().detach(), partial_targets[0], index)
                confidence_txt = utils.confidence_update(confidence_txt, outputs[1].clone().detach(), partial_targets[1], index)

            if epoch >= 0:
                loss.backward()
                optimizer.step()
            train_loss += loss.item()

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | LR: %g' % (train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']))

    def eval(data_loader, epoch, mode = 'test'):
        fea, lab = [[] for _ in range(n_view)], [[] for _ in range(n_view)]
        test_loss, loss_list, correct_list, total_list = 0., [0.] * n_view, [0.] * n_view, [0.] * n_view

        with torch.no_grad():
            if sum([data_loader.dataset.train_data[v].shape[0] != data_loader.dataset.train_data[0].shape[0] for v in range(len(data_loader.dataset.train_data))]) == 0:
                for batch_idx, (batches, partial_targets, true_targets, index) in enumerate(data_loader):

                    batches, partial_targets, true_targets = [batches[v].cuda() for v in range(n_view)], [
                        partial_targets[v].cuda() for v in range(n_view)], [true_targets[v].cuda() for v in range(n_view)]

                    logits0, outputs0 = [multi_models[v](batches[v]) for v in range(n_view)]
                    logits = [logits0[0], outputs0[0]]
                    outputs = [logits0[1], outputs0[1]]

                    pred, losses = [], []
                    for v in range(n_view):
                        fea[v].append(outputs[v])
                        lab[v].append(true_targets[v])
                        pred.append(outputs[v].mm(C))

                        _, predicted = pred[v].max(1)
                        total_list[v] += true_targets[v].size(0)
                        acc = predicted.eq(true_targets[v]).sum().item()
                        correct_list[v] += acc

            else:
                pred, losses = [], []
                for v in range(n_view):
                    count = int(np.ceil(data_loader.dataset.train_data[v].shape[0]) / data_loader.batch_size)
                    for ct in range(count):
                       
                        batch, partial_targets, true_targets = torch.Tensor(data_loader.dataset.train_data[v][ct * data_loader.batch_size: (ct + 1) * data_loader.batch_size]).cuda(), torch.Tensor(data_loader.dataset.partial_label[v][ct * data_loader.batch_size: (ct + 1) * data_loader.batch_size]).long().cuda(), torch.Tensor(data_loader.dataset.true_label[v][ct * data_loader.batch_size: (ct + 1) * data_loader.batch_size]).long().cuda()
                        outputs = multi_models[v](batch)

                        fea[v].append(outputs)
                        lab[v].append(true_targets)
                        pred.append(outputs.mm(C))

                        _, predicted = pred[v].max(1)
                        total_list[v] += true_targets.size(0)
                        acc = predicted.eq(true_targets).sum().item()
                        correct_list[v] += acc

            fea = [torch.cat(fea[v]).cpu().detach().numpy() for v in range(n_view)]
            lab = [torch.cat(lab[v]).cpu().detach().numpy() for v in range(n_view)]
        test_dict = {('view_%d_loss' % v): loss_list[v] / len(data_loader) for v in range(n_view)}
        test_dict['sum_loss'] = test_loss / len(data_loader)
        summary_writer.add_scalars('Loss/' + mode, test_dict, epoch)

        summary_writer.add_scalars('Accuracy/' + mode, {('view_%d_acc' % v): correct_list[v] / total_list[v] for v in range(n_view)}, epoch)

        return fea, lab

    def multiview_test(fea, lab):
        MAPs = np.zeros([n_view, n_view])
        val_dict = {}
        print_str = ''
        for i in range(n_view):
            for j in range(n_view):
                if i == j:
                    continue
                MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                key = '%s2%s' % (args.views[i], args.views[j])
                val_dict[key] = MAPs[i, j]
                print_str = print_str + key + ': %.3f\t' % val_dict[key]
        return val_dict, print_str

    def test(epoch):
            global best_acc
            global best_epoch
            set_eval()

            fea, lab = eval(valid_loader, epoch, 'valid')

            MAPs = np.zeros([n_view, n_view])
            val_dict = {}
            print_val_str = 'Validation: '

            for i in range(n_view):
                for j in range(n_view):
                    if i == j:
                        continue
                    MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                    key = '%s2%s' % (args.views[i], args.views[j])
                    val_dict[key] = MAPs[i, j]
                    print_val_str = print_val_str + key +': %g\t' % val_dict[key]


            val_avg = MAPs.sum() / n_view / (n_view - 1.)
            val_dict['avg'] = val_avg
            print_val_str = print_val_str + 'Avg: %g' % val_avg
            summary_writer.add_scalars('Retrieval/valid', val_dict, epoch)

            fea, lab = eval(test_loader, epoch, 'test')

            MAPs = np.zeros([n_view, n_view])
            test_dict = {}
            print_test_str = 'Test: '
            for i in range(n_view):
                for j in range(n_view):
                    if i == j:
                        continue
                    MAPs[i, j] = fx_calc_map_label(fea[j], lab[j], fea[i], lab[i], k=0, metric='cosine')[0]
                    key = '%s2%s' % (args.views[i], args.views[j])
                    test_dict[key] = MAPs[i, j]
                    print_test_str = print_test_str + key + ': %g\t' % test_dict[key]

            test_avg = MAPs.sum() / n_view / (n_view - 1.)
            print_test_str = print_test_str + 'Avg: %g' % test_avg
            test_dict['avg'] = test_avg
            summary_writer.add_scalars('Retrieval/test', test_dict, epoch)

            print(print_val_str)
            if val_avg > best_acc:

                best_epoch = epoch
                best_acc = val_avg
                print(print_test_str)
                print('Saving..')
                state = {}
                for v in range(n_view):

                    state['model_state_dict_%d' % v] = multi_models[v].state_dict()
                for key in test_dict:
                    state[key] = test_dict[key]
                state['epoch'] = epoch
                state['optimizer_state_dict'] = optimizer.state_dict()
                state['C'] = C
                torch.save(state, os.path.join(args.ckpt_dir, '%s_%s_%d_best_checkpoint.t7' % ('PLL_MRL_CMR', args.data_name, args.output_dim)))
            return val_dict


    tempY = [[], []]
    uniform_confidence = [[],[]]

    prototypes_img = torch.zeros(args.num_class, args.output_dim).cuda()
    prototypes_txt = torch.zeros(args.num_class, args.output_dim).cuda()

    all_partial_targets = train_dataset.partial_label
    all_true_targets = train_dataset.train_label


    if args.method == "ours":
        tempY[0] = [sum(sublist) for sublist in all_partial_targets[0]]
        tempY[1] = [sum(sublist) for sublist in all_partial_targets[1]]

        uniform_confidence[0] = [ item1/item2 for item1,item2 in zip(all_partial_targets[0], tempY[0])]
        uniform_confidence[1] = [item1 / item2 for item1, item2 in zip(all_partial_targets[1], tempY[1])]

        uniform_confidence = torch.tensor(uniform_confidence)
        uniform_confidence = uniform_confidence.cuda()
        confidence_img = uniform_confidence[0].cuda()
        confidence_txt = uniform_confidence[1].cuda()

    logit = [[], []]

    best_prec1 = 0.
    lr_schedu.step(start_epoch)
    train(-1,confidence_img, confidence_txt, all_partial_targets, all_true_targets, prototypes_img, prototypes_txt)
    results = test(-1)
    for epoch in range(start_epoch, args.max_epochs):
        train(epoch, confidence_img, confidence_txt, all_partial_targets, all_true_targets, prototypes_img, prototypes_txt)
        lr_schedu.step(epoch)
        test_dict = test(epoch + 1)
        if test_dict['avg'] == best_acc:
            multi_model_state_dict = [{key: value.clone() for (key, value) in m.state_dict().items()} for m in multi_models]
            W_best = C.clone()

    print('Evaluation on Last Epoch:')
    fea, lab = eval(test_loader, epoch, 'test')
    test_dict, print_str = multiview_test(fea, lab)
    print(print_str)

    print('Evaluation on Best Validation:')
    [multi_models[v].load_state_dict(multi_model_state_dict[v]) for v in range(n_view)]
    fea, lab = eval(test_loader, epoch, 'test')
    test_dict, print_str = multiview_test(fea, lab)
    print(print_str)

    save_dict = dict(**{args.views[v]: fea[v] for v in range(n_view)}, **{args.views[v] + '_lab': lab[v] for v in range(n_view)})
    save_dict['C'] = W_best.detach().cpu().numpy()
    sio.savemat('features/%s_%g.mat' % (args.data_name, args.partial_ratio), save_dict)

def fx_calc_map_multilabel_k(train, train_labels, test, test_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(test, train, metric)
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i].reshape(-1)

        tmp_label = (np.dot(train_labels[order], test_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)

def fx_calc_map_label(train, train_labels, test, test_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(test, train, metric)

    ord = dist.argsort(1)

    numcases = train_labels.shape[0]
    if k == 0:
        k = numcases
    if k == -1:
        ks = [50, numcases]
    else:
        ks = [k]

    def calMAP(_k):
        _res = []
        for i in range(len(test_label)):
            order = ord[i]
            p = 0.0
            r = 0.0
            for j in range(_k):
                if test_label[i] == train_labels[order[j]]:
                    r += 1
                    p += (r / (j + 1))
            if r > 0:
                _res += [p / r]
            else:
                _res += [0]
        return np.mean(_res)

    res = []
    for k in ks:
        res.append(calMAP(k))
    return res

if __name__ == '__main__':
    main()

