import time
import numpy as np
import os.path as osp
from collections import OrderedDict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.tensorboard import SummaryWriter
from utils.utils import AverageMeterGroups
from metrics.psnr_ssim import calculate_psnr
from utils.build_utils import build_from_cfg


class Trainer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = self.config['local_rank']
        self._init_dataset()
        self._init_loss()
        self.model_name = config['exp_name']
        self.model = build_from_cfg(config.network).to(self.config.device)
        
        # 冻结gmsf层的参数
        for param in self.model.particle_flow.parameters():
            param.requires_grad = False

        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=config.lr, weight_decay=config.weight_decay)
        
        if config['distributed']:
            self.model = DDP(self.model,
                             device_ids=[self.rank],
                             output_device=self.rank,
                             broadcast_buffers=True,
                             find_unused_parameters=False)
        self.resume_training()
    
    def resume_training(self):
        ckpt_path = self.config.get('resume_state')
        if ckpt_path is not None:
            ckpt = torch.load(self.config['resume_state'])
            if self.config['distributed']:
                self.model.module.load_state_dict(ckpt['state_dict'])
            else:
                self.model.load_state_dict(ckpt['state_dict'])
            print("finetune from: ", ckpt_path)
            self.resume_epoch = 0
        else:
            self.resume_epoch = 0

    def _init_dataset(self):
        dataset_train = build_from_cfg(self.config.data.train)
        dataset_val = build_from_cfg(self.config.data.val)
         
        self.sampler = DistributedSampler(
            dataset_train, num_replicas=self.config['world_size'], rank=self.config['local_rank'])
        self.config.data.train_loader.batch_size //= self.config['world_size']
        self.loader_train = DataLoader(dataset_train,
                                       **self.config.data.train_loader,
                                       pin_memory=True, drop_last=True, sampler=self.sampler)

        self.loader_val = DataLoader(dataset_val, **self.config.data.val_loader,
                                     pin_memory=True, shuffle=False, drop_last=False)

    def _init_loss(self):
        self.loss_dict = dict()
        for loss_cfg in self.config.losses:
            loss = build_from_cfg(loss_cfg)
            self.loss_dict[loss_cfg['nickname']] = loss

    def set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self, iters):
        ratio = 0.5 * (1.0 + np.cos(iters /
                                    (self.config['epochs'] * self.loader_train.__len__()) * np.pi))
        lr = (self.config['lr'] - self.config['lr_min']
              ) * ratio + self.config['lr_min']
        return lr

    def train(self):
        local_rank = self.config['local_rank']
        log_path = osp.join('log', self.config['exp_name'])
        if local_rank <= 0:
            writer = SummaryWriter(log_path + '/train')
            writer_val = SummaryWriter(log_path + '/validate')
        else:
            writer, writer_val = None, None

        best_psnr = 0.0
        loss_group = AverageMeterGroups()
        time_group = AverageMeterGroups()
        iters_per_epoch = self.loader_train.__len__()
        iters = self.resume_epoch * iters_per_epoch
        total_iters = self.config['epochs'] * iters_per_epoch

        psnr, eval_t = self.evaluate(iters, writer_val)
        best_psnr = psnr
        print('ori_psnr: ', best_psnr)
        self.save('psnr_best.pth', 0)

        start_t = time.time()
        total_t = 0
        for epoch in range(self.resume_epoch, self.config['epochs']):
            self.sampler.set_epoch(epoch)
            for data in self.loader_train:
                for k, v in data.items():
                    data[k] = v.to(self.config['device'])
                data_t = time.time() - start_t

                lr = self.get_lr(iters)
                self.set_lr(self.optimizer, lr)

                self.optimizer.zero_grad()
                results = self.model(**data)
                total_loss = torch.tensor(0., device=self.config['device'])
                for name, loss in self.loss_dict.items():
                    l = loss(**results, **data)
                    loss_group.update({name: l.cpu().data})
                    total_loss += l
                total_loss.backward()
                self.optimizer.step()

                iters += 1

                iter_t = time.time() - start_t
                total_t += iter_t
                time_group.update({'data_t': data_t, 'iter_t': iter_t})

                if (iters+1) % 20 == 0 and local_rank == 0:
                    tpi = total_t / (iters - self.resume_epoch * iters_per_epoch)
                    eta = total_iters * tpi
                    remainder = (total_iters - iters) * tpi
                    eta = self.eta_format(eta)
                    writer.add_scalar("loss", total_loss, iters)
                    remainder = self.eta_format(remainder)
                    print('epoch:{} iters:{} time:{:.2f} loss:{:.3f}'.format(epoch, iters, iter_t, total_loss))
                    loss_group.reset()
                    time_group.reset()
                start_t = time.time()

            if (epoch+1) % self.config['eval_interval'] == 0 and local_rank == 0:
                psnr, eval_t = self.evaluate(iters, writer_val)
                writer_val.add_scalar("psnr", psnr, iters)
                total_t += eval_t
                if psnr > best_psnr:
                    best_psnr = psnr
                    self.save('psnr_best.pth', epoch)
                if (epoch+1) % 50 == 0:
                    self.save(f'epoch_{epoch+1}.pth', epoch)
                self.save('latest.pth', epoch)

    def evaluate(self, iters, writer_val):
        psnr_list = []
        time_stamp = time.time()
        for i, data in enumerate(self.loader_val):
            for k, v in data.items():
                data[k] = v.to(self.config['device'])

            with torch.no_grad():
                results = self.model(**data, eval=True)
                imgt_pred = results['imgt_pred']
                for j in range(data['img0'].shape[0]):
                    psnr = calculate_psnr(imgt_pred[j].detach().unsqueeze(
                        0), data['imgt'][j].unsqueeze(0))
                    psnr_list.append(psnr)
                if i==0: 
                    gt = (data['imgt'][j].permute(1,2,0).detach().cpu().numpy() * 255).astype('uint8')
                    pred = (imgt_pred[j].permute(1,2,0).detach().cpu().numpy() * 255).astype('uint8')
                    for k in range(1):
                        img = np.concatenate((gt[:,:,3*k:3*k+3], pred[:,:,3*k:3*k+3]), 1)
                        writer_val.add_image(str(k) + '/img', img, iters, dataformats='HWC')

        eval_time = time.time() - time_stamp

        print('eval iter:{} time:{:.2f} psnr:{:.3f}'.format(
            iters, eval_time, np.array(psnr_list).mean()))
        return np.array(psnr_list).mean(), eval_time

    def save(self, name, epoch):
        save_path = '{}/{}/{}'.format(self.config['save_dir'], 'ckpts', name)
        ckpt = OrderedDict(epoch=epoch)
        if self.config['distributed']:
            ckpt['state_dict'] = self.model.module.state_dict()
        else:
            ckpt['state_dict'] = self.model.state_dict()
        ckpt['optim'] = self.optimizer.state_dict()
        torch.save(ckpt, save_path)

    def eta_format(self, eta):
        time_str = ''
        if eta >= 3600:
            hours = int(eta // 3600)
            eta -= hours * 3600
            time_str = f'{hours}'

        if eta >= 60:
            mins = int(eta // 60)
            eta -= mins * 60
            time_str = f'{time_str}:{mins:02}'

        eta = int(eta)
        time_str = f'{time_str}:{eta:02}'
        return time_str
