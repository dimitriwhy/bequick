#!/usr/bin/env python
import platform
import argparse
import logging
import numpy as np
import tensorflow as tf
import copy
import threading
from multiprocessing import Process, Manager, Queue
try:
    import bequick
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from bequick.corpus import read_conllx_dataset, get_alphabet
from bequick.embedding import load_embedding
try:
    from .tb_parser import TransitionSystem, Parser, State
    from .model import Classifier, DeepQNetwork, initialize_word_embeddings
    from .tree_utils import is_projective_raw, is_tree_raw
    from .evaluate import evaluate
    from .instance_builder import InstanceBuilder
except (ValueError, SystemError) as e:
    from tb_parser import TransitionSystem, Parser, State
    from model import Classifier, DeepQNetwork, initialize_word_embeddings
    from tree_utils import is_projective_raw, is_tree_raw
    from evaluate import evaluate
    from instance_builder import InstanceBuilder

np.random.seed(1234)
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
LOG = logging.getLogger('chen2014')

def find_best(parser, state, scores):
    best_score, best_i, best_name = None, None, None
    for i, score in enumerate(scores):
        name = parser.get_action(i)
        if state.valid(name) and (best_score is None or score > best_score):
            best_score, best_i, best_name = score, i, name
    return best_score, best_i, best_name


def get_valid_actions(parser, state):
    """
        
        :param parser: Parser
        :param state: State
        :return: tuple
        """
    n_actions = parser.system.num_actions()
    aids = []
    mask = np.zeros(n_actions, dtype=np.bool)
    for aid in range(n_actions):
        if parser.system.valid(state, aid):
            aids.append(aid)
            mask[aid] = True
    return aids, mask

def calc_prob_distribution(prediction):
    value_sum = np.float32(0)
    max_sum = np.NINF
    for aid in range(len(prediction)):
        if prediction[aid] > max_sum:
            max_sum = prediction[aid]
    for aid in range(len(prediction)):
        if prediction[aid] != np.NINF:
            prediction[aid] = np.exp(prediction[aid] - max_sum)
            value_sum += prediction[aid]
    for aid in range(len(prediction)):
        if prediction[aid] != np.NINF:
            prediction[aid] /= value_sum
            
def choose_action(prediction):
    chosen_id = -1
    p = np.random.rand()
    for aid in range(len(prediction)):
        if prediction[aid] != np.NINF:
            chosen_id = aid
            p -= prediction[aid]
            if p <= np.float32(0) :
                break
    return chosen_id

def discount_future_rewards(rewards):
    future_rewards = [0 for i in range(len(rewards))]
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * opts.discount_factor + rewards[t]
        future_rewards[t] = running_add
    return future_rewards


def supervised_learning(model, session):
    best_uas = 0.
    forms, postags, deprels, Ys = [], [], [], []
    for n, train_data in enumerate(train_dataset):
        xs, ys = parser.generate_training_instance(train_data)
        forms.append(xs[0])
        postags.append(xs[1])
        deprels.append(xs[2])
        Ys.append(ys)
    forms = np.concatenate(forms)
    postags = np.concatenate(postags)
    deprels = np.concatenate(deprels)
    Ys = np.concatenate(Ys)

    n_batch, n_samples = 0, Ys.shape[0]
    test_uas, test_las = None, None
    order = np.arange(n_samples)
    LOG.info('Training sample size: {0}'.format(n_samples))
    for i in range(1, opts.max_iter + 1):
        np.random.shuffle(order)
        cost = 0.
        for batch_start in range(0, n_samples, opts.batch_size):
            batch_end = batch_start + opts.batch_size if batch_start + opts.batch_size < n_samples else n_samples
            batch_id = order[batch_start: batch_end]
            xs, ys = (forms[batch_id], postags[batch_id], deprels[batch_id]), Ys[batch_id]
            cost += model.train(session, xs, ys)
            n_batch += 1
            if opts.evaluate_stops > 0 and n_batch % opts.evaluate_stops == 0:
                uas, las = evaluate(devel_dataset, session, parser, model, devel_feats, True, opts.lang)
                LOG.info('At {0}, UAS={1}, LAS={2}'.format((float(n_batch) / n_samples), uas, las))
                if uas > best_uas:
                    best_uas, best_las = uas, las
                    test_uas, test_las = evaluate(test_dataset, session, parser, model, test_feats, True, opts.lang)
                    LOG.info('New best achieved: {0}, test: UAS={1}, LAS={2}'.format(best_uas, test_uas, test_las ))
        uas, las = evaluate(devel_dataset, session, parser, model, devel_feats, True, opts.lang)
        LOG.info('Iteration {0} done, Cost={1}, UAS={2}, LAS={3}'.format(i, cost, uas, las))
        if uas > best_uas:
            best_uas, best_las = uas, las
            test_uas, test_las = evaluate(test_dataset, session, parser, model, test_feats, True, opts.lang)
            LOG.info('New best achieved: {0}, test: UAS={1}, LAS={2}'.format(best_uas, test_uas, test_las))
            model.save(session)
            LOG.info('New best result is saved.')
    LOG.info('Finish training, best devel UAS: {0}, test UAS={1}, LAS={2}'.format(best_uas, test_uas, test_las))
    model.restore(session)
    LOG.info('Restore best result from file.')
    uas, las = evaluate(devel_dataset, session, parser, model, devel_feats, True, opts.lang)
    test_uas, test_las = evaluate(test_dataset, session, parser, model, test_feats, True, opts.lang)
    LOG.info('Best devel UAS: {0}, test UAS={1}, LAS{2}'.format(uas, test_uas, test_las))
    
if __name__ == "__main__":
    cmd = argparse.ArgumentParser("An implementation of Chen and Manning (2014)'s parser")
    # Arguments for supervised learning
    cmd.add_argument("--embedding", help="The path to the embedding file.")
    cmd.add_argument("--reference", help="The path to the reference file.")
    cmd.add_argument("--development", help="The path to the development file.")
    cmd.add_argument("--test", help="The path to the test file.")
    cmd.add_argument("--init-range", dest="init_range", type=float, default=0.01, help="The initialization range.")
    cmd.add_argument("--max-iter", dest="max_iter", type=int, default=200,
                     help="The number of max iteration for supervised learning.")
    cmd.add_argument("--hidden-size", dest="hidden_size", type=int, default=400, help="The size of hidden layer.")
    cmd.add_argument("--embedding-size", dest="embedding_size", type=int, default=100, help="The size of embedding.")
    cmd.add_argument("--evaluate-stops", dest="evaluate_stops", type=int, default=-1, help="Evaluate on per n iters.")
    cmd.add_argument("--ada-eps", dest="ada_eps", type=float, default=1e-6, help="The EPS in AdaGrad.")
    cmd.add_argument("--ada-alpha", dest="ada_alpha", type=float, default=0.01, help="The Alpha in AdaGrad.")
    cmd.add_argument("--lambda", dest="lamb", type=float, default=1e-8, help="The regularizer parameter.")
    cmd.add_argument("--batch-size", dest="batch_size", type=int, default=5000, help="The size of batch.")
    cmd.add_argument("--dropout", type=float, default=0.5, help="The probability for dropout.")
    cmd.add_argument("--language", dest="lang", default="en", help="the language")
    #Argument for reinforcement learning
    cmd.add_argument("--batch-size-rl", dest="batch_size_rl", type=int, default=3000, help="The size of batch for rl.")
    cmd.add_argument("--max-iter-rl", dest="max_iter_rl", type=int, default=300, help="The number of max iteration for rl.")
    cmd.add_argument("--mc-steps", dest="mc_steps", type=int, default=10, help="The number of max iteration for MC.")
    cmd.add_argument("--discount-factor", dest="discount_factor", type=float, default=0.99, help="The discount factor for RL.")
    cmd.add_argument("--fault-factor", dest="fault_factor", type=float, default = -50, help="The fault factor for policy selection.")
    
    opts = cmd.parse_args()

    raw_train = read_conllx_dataset(opts.reference)
    raw_devel = read_conllx_dataset(opts.development)
    raw_test = read_conllx_dataset(opts.test)
    LOG.info("Dataset stats: #train={0}, #devel={1}, #test={2}.".format(len(raw_train), len(raw_devel), len(raw_test)))
    raw_train = [data for data in raw_train if is_tree_raw(data) and is_projective_raw(data)]
    LOG.info("{0} training sentences after filtering non-tree and non-projective.".format(len(raw_train)))

    form_alphabet = get_alphabet(raw_train, 'form')
    pos_alphabet = get_alphabet(raw_train, 'pos')
    deprel_alphabet = get_alphabet(raw_train, 'deprel')
    form_alphabet['_ROOT_'] = len(form_alphabet)
    pos_alphabet['_ROOT_'] = len(pos_alphabet)
    LOG.info("Alphabet stats: #form={0} (w/ nil & unk & root), #pos={1} (w/ nil & unk & root),"
             " #deprel={2} (w/ nil & unk)".format(len(form_alphabet), len(pos_alphabet), len(deprel_alphabet)))

    instance_builder = InstanceBuilder(form_alphabet, pos_alphabet, deprel_alphabet)
    train_dataset = instance_builder.conllx_to_instances(raw_train, add_pseudo_root=True)
    devel_dataset = instance_builder.conllx_to_instances(raw_devel, add_pseudo_root=True)
    test_dataset = instance_builder.conllx_to_instances(raw_test, add_pseudo_root=True)
    devel_feats = [[None] + [token['feat'] for token in data] for data in raw_devel]
    test_feats = [[None] + [token['feat'] for token in data] for data in raw_test]
    LOG.info("Dataset converted from string to index.")
    LOG.info("{0} training sentences after filtering non-tree and non-projective.".format(len(train_dataset)))

    system = TransitionSystem(deprel_alphabet)
    parser = Parser(system)

    model = Classifier(form_size=len(form_alphabet), form_dim=100, pos_size=len(pos_alphabet), pos_dim=20,
                       deprel_size=len(deprel_alphabet), deprel_dim=20, hidden_dim=opts.hidden_size,
                       output_dim=system.num_actions(), dropout=opts.dropout, l2=opts.lamb)
    indices, matrix = load_embedding(opts.embedding, form_alphabet, opts.embedding_size)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.inter_op_parallelism_threads = 8
    config.intra_op_parallelism_threads = 8
    session = tf.Session(config = config)
    session.run(tf.global_variables_initializer())
    initialize_word_embeddings(session, model.form_emb, indices, matrix)
    LOG.info('Embedding is loaded.')

    supervised_learning(model, session)

    n, n_batch, iteration = 0, 0, 0
    best_uas, test_uas, test_las = 0., 0., 0.
    n_actions = parser.system.num_actions()

    total_forms, total_postags, total_deprels, total_Ys, total_rewards = [], [], [], [], []
    while iteration <= opts.max_iter_rl:
        if n == 0:
            cost = 0
            logging.info("Start of policy gradient iteration {0}, data shuffled.".format(iteration))
            np.random.shuffle(train_dataset)
        data = train_dataset[n]

        s = State(data)
        forms, postags, deprels, Ys, rewards = [], [], [], [], []
        fault_forms, fault_postags, fault_deprels, fault_Ys, fault_rewards = [], [], [], [], []
        score = 0
        prob = []
        while not s.terminate():
            x = parser.parameterize_x(s)
            prediction = model.classify(session, x)[0]
            valid_ids, valid_mask = get_valid_actions(parser, s)
            #if n == 27:
            # LOG.info("valid_ids = {0}\nvalid_mask={1}".format(valid_ids, valid_mask))
            #prediction[~valid_mask] = np.NINF
            # LOG.info("raw_prediction = {0}".format(prediction))
            calc_prob_distribution(prediction)
            # LOG.info("prediction = {0}".format(prediction))
            while True:
                chosen_id = choose_action(prediction)
                if valid_mask[chosen_id]:
                    break
                fault_forms.append(x[0])
                fault_postags.append(x[1])
                fault_deprels.append(x[2])
                fault_Ys.append(chosen_id)
                fault_rewards.append(opts.fault_factor)

            # prob.append(prediction[chosen_id])
            forms.append(x[0])
            postags.append(x[1])
            deprels.append(x[2])
            Ys.append(chosen_id)
            r = system.scored_transit(s, chosen_id)
            rewards.append(r)
            score += r

        # LOG.info("prob = {0} \n rewards = {1} \n".format(prob, rewards))
        
        forms += fault_forms
        postags += fault_postags
        deprels += fault_deprels
        Ys += fault_Ys

        total_forms += forms
        total_postags += postags
        total_deprels += deprels
        total_Ys += Ys

        future_rewards = discount_future_rewards(rewards)
        future_rewards += fault_rewards
        total_rewards += future_rewards
        # LOG.info("score = {0}".format(score))
        
        # future_rewards -= np.mean(future_rewards)
        # future_rewards /= np.std(future_rewards)
        if len(total_rewards) >= opts.batch_size_rl :
            while len(total_rewards) > opts.batch_size_rl+100:
                forms = np.concatenate(total_forms[:opts.batch_size_rl+100])
                postags = np.concatenate(total_postags[:opts.batch_size_rl+100])
                deprels = np.concatenate(total_deprels[:opts.batch_size_rl+100])
                Ys = np.array(total_Ys[:opts.batch_size_rl+100])
                future_rewards = np.array(total_rewards[:opts.batch_size_rl+100])
                if len(forms) & len(postags) & len(deprels) & len(Ys) & len(future_rewards) \
                   != len(future_rewards) :
                    LOG.info("fuck")
                else:
                    LOG.info("ok")
                    
                cost += model.policy_train(session, (forms, postags, deprels), Ys, future_rewards)

                total_forms = total_forms[opts.batch_size_rl+100:]
                total_postags = total_postags[opts.batch_size_rl+100:]
                total_deprels = total_deprels[opts.batch_size_rl+100:]
                total_Ys = total_Ys[opts.batch_size_rl+100:]
                total_rewards = total_rewards[opts.batch_size_rl+100:]
                
            forms = np.concatenate(total_forms)
            postags = np.concatenate(total_postags)
            deprels = np.concatenate(total_deprels)
            Ys = np.array(total_Ys)
            future_rewards = np.array(total_rewards)
            LOG.info("trainning batch size = {0}".format(len(future_rewards)))
            cost += model.policy_train(session, (forms, postags, deprels), Ys, future_rewards)
            total_forms, total_postags, total_deprels, total_Ys, total_rewards = [], [], [], [], []
            
            # LOG.info("train")
            
        # MOVE to the next sentence
        n += 1
        if n == len(train_dataset):
            uas, las = evaluate(devel_dataset, session, parser, model, devel_feats, True, opts.lang)
            if n == len(train_dataset):
                LOG.info('Iteration {0} done, Cost={1}, UAS={2}, LAS={3}'.format(iteration, cost,
                                                                                          uas, las))
                iteration += 1
                n = 0
            if uas > best_uas:
                best_uas = uas
                test_uas, test_las = evaluate(test_dataset, session, parser, model, test_feats, True, opts.lang)
                LOG.info('New best achieved: {0}, test: UAS={1}, LAS={2}'.format(best_uas, test_uas, test_las))
        if n % 2000 == 0:
            LOG.info('Partly finished at {0}'.format(n))
        

    LOG.info('Finish training, best devel UAS={0}, test UAS={1}, LAS={2}'.format(best_uas, test_uas, test_las))
