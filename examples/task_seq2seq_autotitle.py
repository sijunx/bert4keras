#! -*- coding: utf-8 -*-
# bert做Seq2Seq任务，采用UNILM方案
# 介绍链接：https://kexue.fm/archives/6933

from __future__ import print_function
import glob
import os

import numpy as np
import sys

from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model, tf
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from examples.my_args import arg_dic

from keras import backend as K
from tensorflow.python.platform import gfile

# parameter ==========================
wkdir = '/Users/xusijun/Documents/NLP009/bert4keras-master001/keras_to_tensorflow-master'
pb_filename = 'model.pb'

# 基本参数
maxlen = 256
batch_size = 16
# steps_per_epoch = 1000
steps_per_epoch = 5
# epochs = 10000
epochs = 2

# bert配置
# config_path = '/root/kg/bert/chinese_wwm_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/root/kg/bert/chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '/root/kg/bert/chinese_wwm_L-12_H-768_A-12/vocab.txt'

# config_path = '/Users/xusijun/Documents/NLP009/bert4keras-master001/chinese_wwm_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/Users/xusijun/Documents/NLP009/bert4keras-master001/chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '/Users/xusijun/Documents/NLP009/bert4keras-master001/chinese_wwm_L-12_H-768_A-12/vocab.txt'

config_path = '/Users/xusijun/Documents/NLP009/bert4keras-master001/albert_tiny_google_zh_489k/albert_config.json'
checkpoint_path = '/Users/xusijun/Documents/NLP009/bert4keras-master001/albert_tiny_google_zh_489k/albert_model.ckpt'
dict_path = '/Users/xusijun/Documents/NLP009/bert4keras-master001/albert_tiny_google_zh_489k/vocab.txt'

# 训练样本。THUCNews数据集，每个样本保存为一个txt。
# txts = glob.glob('/root/thuctc/THUCNews/*/*.txt')
txts = glob.glob('/Users/xusijun/Documents/NLP009/bert4keras-master001/MyNews/*/*.txt')

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, txt in self.sample(random):
            text = open(txt, encoding='utf-8').read()
            text = text.split('\n')
            if len(text) > 1:
                title = text[0]
                content = '\n'.join(text[1:])
                token_ids, segment_ids = tokenizer.encode(
                    content, title, maxlen=maxlen
                )
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(model).predict([token_ids, segment_ids])

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)

# save model to pb ====================
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def my_keras_to_pb():
    # save keras model as tf pb files ===============
    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, wkdir, pb_filename, as_text=False)

    # # load & inference the model ==================
    with tf.Session() as sess:
        # load model from pb file
        with gfile.FastGFile(wkdir+'/'+pb_filename,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            g_in = tf.import_graph_def(graph_def)
        # write to tensorboard (check tensorboard for each op names)
        writer = tf.summary.FileWriter(wkdir+'/log/')
        writer.add_graph(sess.graph)
        writer.flush()
        writer.close()
        # print all operation names
        print('\n===== ouptut operation names =====\n')
        for op in sess.graph.get_operations():
          print(op)
        # inference by the model (op name must comes with :0 to specify the index of its output)
        tensor_output = sess.graph.get_tensor_by_name('cross_entropy_1/Identity:0')
        tensor_input = sess.graph.get_tensor_by_name('Input-Token:0')
        tokens_a = tokenizer.tokenize('测试数据')
        predictions = sess.run(tensor_output, {tensor_input: tokens_a})
        print('\n===== output predicted results =====\n')
        print(predictions)
        print('xxxxxxxxxx')

def just_show():
    s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及 时 就医 。'
    s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午 ，华 住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'
    for s in [s1, s2]:
        print(u'生成标题:', autotitle.generate(s))
    print()

def create_classification_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels):
    # 通过传入的训练数据，进行representation
    # model = model.BertModel(
    #     config=bert_config,
    #     is_training=is_training,
    #     input_ids=input_ids,
    #     input_mask=input_mask,
    #     token_type_ids=segment_ids,
    # )

    embedding_layer = model.get_sequence_output()
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        if labels is not None:
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
        else:
            loss, per_example_loss = None, None
    return (loss, per_example_loss, logits, probabilities)

def save_PBmodel( num_labels):
    """    保存PB格式中文分类模型    """
    try:
        # 如果PB文件已经存在，则返回PB文件的路径，否则将模型转化为PB文件，并且返回存储PB文件的路径
        pb_file = os.path.join(arg_dic['pb_model_dir'], 'classification_model.pb')
        graph = tf.Graph()
        with graph.as_default():
            input_ids = tf.placeholder(tf.int32, (None, arg_dic['max_seq_length']), 'input_ids')
            input_mask = tf.placeholder(tf.int32, (None, arg_dic['max_seq_length']), 'input_mask')
            # bert_config = modeling.BertConfig.from_json_file(arg_dic['bert_config_file'])
            bert_config = model.BertConfig.from_json_file(arg_dic['bert_config_file'])
            loss, per_example_loss, logits, probabilities = create_classification_model(
                bert_config=bert_config, is_training=False,
                input_ids=input_ids, input_mask=input_mask, segment_ids=None, labels=None, num_labels=num_labels)

            probabilities = tf.identity(probabilities, 'pred_prob')
            saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                latest_checkpoint = tf.train.latest_checkpoint(arg_dic['output_dir'])
                saver.restore(sess, latest_checkpoint)
                from tensorflow.python.framework import graph_util
                tmp_g = graph_util.convert_variables_to_constants(sess, graph.as_graph_def(), ['pred_prob'])

        # 存储二进制模型到文件中
        with tf.gfile.GFile(pb_file, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return pb_file
    except Exception as e:
        print('fail to optimize the graph! %s', e)

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            # model.save_weights('./best_model.weights')
            # model.save_weights('/Users/xusijun/Documents/NLP009/bert4keras-master001/tensorflow-for-java-master/best_model03.weights')
            # model.save('./myFilename.h5')
            # model.save('/Users/xusijun/Documents/NLP009/bert4keras-master001/tensorflow-for-java-master/myFile04')
        # 演示效果
        just_show()
        # 我的保存
        my_keras_to_pb()

if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(txts, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./best_model.weights')
