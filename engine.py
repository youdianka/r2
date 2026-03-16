import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
import torch.nn.functional as F

device = torch.device('cuda', 1)


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    logger,
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train()
    transform = transforms.Compose([
        transforms.Resize(224),
    ])
    loss_list = []

    for iter, data in enumerate(train_loader):
        # 修正步数计算，每个批次加1
        step += 1

        optimizer.zero_grad()
        images = data['image'].to(device, dtype=torch.float)
        targets = data['mask'].to(device, dtype=torch.float32)
        images = transform(images)
        targets = transform(targets)

        # 前向传播
        outputs = model(images)

        # 处理可能的元组输出
        if isinstance(outputs, tuple):
            pred_output = outputs[0]  # 使用第一个输出作为预测
        else:
            pred_output = outputs

        # 计算损失
        loss = criterion(pred_output, targets)

        # 反向传播
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        # 获取当前学习率
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        # 记录损失
        writer.add_scalar('loss', loss, global_step=step)

        # 打印训练信息
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)

    # 更新学习率
    scheduler.step()
    return step


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    transform = transforms.Compose([
        transforms.Resize(224),
    ])
    loss_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            img = data['image'].to(device, dtype=torch.float)
            msk = data['mask'].to(device, dtype=torch.float32)
            msk = transform(msk)
            img = transform(img)

            # 获取模型输出
            out = model(img)

            # 处理可能的元组输出
            if isinstance(out, tuple):
                pred_output = out[0]  # 使用第一个输出作为预测
            else:
                pred_output = out

            # 计算损失
            loss = criterion(pred_output, msk)
            loss_list.append(loss.item())

            # 收集真实标签
            gts.append(msk.squeeze(1).cpu().detach().numpy())

            # 收集预测结果
            pred_numpy = pred_output.squeeze(1).cpu().detach().numpy()
            preds.append(pred_numpy)

        # 计算评估指标
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch},loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    transform = transforms.Compose([
        transforms.Resize(224),
    ])

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img = data['image'].to(device, dtype=torch.float)
            msk = data['mask'].to(device, dtype=torch.float32)
            msk = transform(msk)
            img = transform(img)

            # 获取模型输出
            out = model(img)

            # 处理可能的元组输出
            if isinstance(out, tuple):
                pred_output = out[0]  # 使用第一个输出作为预测
            else:
                pred_output = out

            # 计算损失
            loss = criterion(pred_output, msk)
            loss_list.append(loss.item())

            # 收集真实标签
            msk_numpy = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk_numpy)

            # 收集预测结果
            pred_numpy = pred_output.squeeze(1).cpu().detach().numpy()
            preds.append(pred_numpy)

            # 保存图像
            # if i % config.save_interval == 0:
            #     save_imgs(img, msk_numpy, pred_numpy, i, config.work_dir + 'outputs/', config.datasets,
            #               config.threshold, test_data_name=test_data_name)
            save_imgs(img, msk_numpy, pred_numpy, i, '/userA02/lyp/seg1/CrossNet_ManbaAndRes/output_ph2/', config.datasets, config.threshold, test_data_name=test_data_name)
         # 计算评估指标
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)

        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)

