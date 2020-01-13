from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):

    # Load data
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    opt.fs_index = loader.fs_index

    # Tensorboard summaries (they're great!)
    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    # Load pretrained model, info file, histories file
    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(
                os.path.join(opt.checkpoint_path, 'infos_' + opt.id + format(int(opt.start_from), '04') + '.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            #need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            need_be_same = ["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[
                    checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(
                os.path.join(opt.checkpoint_path, 'histories_' + opt.id + format(int(opt.start_from), '04') + '.pkl')):
            with open(os.path.join(opt.checkpoint_path,
                                   'histories_' + opt.id + format(int(opt.start_from), '04') + '.pkl')) as f:
                histories = cPickle.load(f)
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opt).cuda()
    # dp_model = torch.nn.DataParallel(model)
    # dp_model = torch.nn.DataParallel(model, [0,2,3])
    dp_model = model

    # Loss function
    update_lr_flag = True
    # Assure in training mode
    dp_model.train()
    parameters = model.named_children()
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    # Optimizer and learning rate adjustment flag
    optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(
            os.path.join(opt.checkpoint_path, 'optimizer' + opt.id + format(int(opt.start_from), '04') + '.pth')):
        optimizer.load_state_dict(torch.load(os.path.join(
            opt.checkpoint_path, 'optimizer' + opt.id + format(int(opt.start_from), '04') + '.pth')))

    optimizer.zero_grad()
    accumulate_iter = 0
    train_loss = 0
    reward = np.zeros([1, 1])
    reset_optimzer_index = 1

    # Training loop
    while True:
        if opt.self_critical_after != -1 and epoch >= opt.self_critical_after and reset_optimzer_index :
            opt.learning_rate_decay_start = opt.self_critical_after
            opt.learning_rate_decay_rate = opt.learning_rate_decay_rate_rl
            opt.learning_rate_decay_every = opt.learning_rate_decay_every_rl
            opt.learning_rate = opt.learning_rate_rl
            reset_optimzer_index = 0

        # Update learning rate once per epoch
        if update_lr_flag:

            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)

            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            update_lr_flag = False
                
        # Load data from train split (0)
        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch(opt.train_split)
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        # Unpack data
        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp
        
        # Forward pass and loss
        optimizer.zero_grad()
        if not sc_flag:
            loss = crit(dp_model(fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:])
        else:
            gen_result, sample_logprobs = dp_model(fc_feats, att_feats, att_masks, opt={'sample_max':0}, mode='sample')
            reward = get_self_critical_reward(dp_model, fc_feats, att_feats, att_masks, data, gen_result, opt)
            loss = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())

        # Backward pass
        accumulate_iter = accumulate_iter + 1
        loss = loss / opt.accumulate_number
        loss.backward()
        if accumulate_iter % opt.accumulate_number == 0:
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            iteration += 1
            accumulate_iter = 0
            train_loss = loss.item() * opt.accumulate_number
            end = time.time()

            if not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(iteration, epoch, train_loss, end - start))
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                      .format(iteration, epoch, np.mean(reward[:, 0]), end - start))

        torch.cuda.synchronize()

        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0) and (iteration!=0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration)

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # Validate and save model 
        if (iteration % opt.save_checkpoint_every == 0):

            # # Evaluate model
            # eval_kwargs = {'split': 'val',
            #                 'dataset': opt.input_json}
            # eval_kwargs.update(vars(opt))
            # val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, crit, loader, eval_kwargs)
            #
            # # Write validation result into summary
            # add_summary_value(tb_summary_writer, 'validation loss', val_loss, iteration)
            # if lang_stats is not None:
            #     for k,v in lang_stats.items():
            #         add_summary_value(tb_summary_writer, k, v, iteration)
            # val_result_history[iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}
            #
            # # Our metric is CIDEr if available, otherwise validation loss
            # if opt.language_eval == 1:
            #     current_score = lang_stats['CIDEr']
            # else:
            #     current_score = - val_loss
            current_score = 0

            # Save model in checkpoint path 
            best_flag = False
            if True:  # if true
                save_id = iteration / opt.save_checkpoint_every
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path,
                                               'model' + opt.id + format(int(save_id), '04') + '.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path,
                                              'optimizer' + opt.id + format(int(save_id), '04') + '.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + format(int(save_id), '04') + '.pkl'),
                          'wb') as f:
                    cPickle.dump(infos, f)
                with open(
                        os.path.join(opt.checkpoint_path, 'histories_' + opt.id + format(int(save_id), '04') + '.pkl'),
                        'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

            # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu)
# memory_pool = torch.FloatTensor(15000, 3, 400, 200).cuda()
# del memory_pool
train(opt)
