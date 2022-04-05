#! -*- coding:utf-8 -*-
# 通过对抗训练增强模型的泛化性能
# 比CLUE榜单公开的同数据集上的BERT base的成绩高2%
# 数据集：IFLYTEK' 长文本分类 (https://github.com/CLUEbenchmark/CLUE)
# 博客：https://kexue.fm/archives/7234
# 适用于Keras 2.3.1

import json
import numpy as np
from bert4keras.backend import keras, search_layer, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from tqdm import tqdm

from bert4keras.models import build_transformer_model, tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from keras import backend as K
from tensorflow.python.platform import gfile

wkdir = '/Users/xusijun/Documents/NLP010/bert4keras-master_bert_060/examples'
pb_filename = 'model_classify_1009000.pb'

num_classes = 3
maxlen = 128
batch_size = 32

# BERT base
# config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

config_path = '/Users/xusijun/Documents/NLP009/bert4keras-master001/albert_tiny_google_zh_489k/albert_config.json'
checkpoint_path = '/Users/xusijun/Documents/NLP009/bert4keras-master001/albert_tiny_google_zh_489k/albert_model.ckpt'
dict_path = '/Users/xusijun/Documents/NLP009/bert4keras-master001/albert_tiny_google_zh_489k/vocab.txt'

def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename) as f:
        for i, l in enumerate(f):
            l = json.loads(l)
            text, label = l['sentence'], l['label']
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data(
    # '/root/CLUE-master/baselines/CLUEdataset/iflytek/train.json'
    '/Users/xusijun/Documents/NLP010/bert4keras-master_bert_060/examples/CLUEdataset/iflytek/train.json'
)
valid_data = load_data(
    # '/root/CLUE-master/baselines/CLUEdataset/iflytek/dev.json'
    '/Users/xusijun/Documents/NLP010/bert4keras-master_bert_060/examples/CLUEdataset/iflytek/dev.json'
)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),
    metrics=['sparse_categorical_accuracy'],
)


def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


# 写好函数后，启用对抗训练只需要一行代码
adversarial_training(model, 'Embedding-Token', 0.5)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


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
        # print('\n===== ouptut operation names =====\n')
        # for op in sess.graph.get_operations():
        #   print(op)

def just_show():
    s1 = u"亚洲金融危机，楼市泡沫，通货膨胀"
    s2 = u"朝闻天下，足不出户知天下"
    s3 = u"羽毛球比赛一直是热门比赛"
    s4 = u"中国男篮世锦赛，小组取得第一名"
    s5 = u"中国男足亚洲杯小组都没有出现"
    s6 = u"经济如何脱实向虚"
    s7 = u"实体经济将越来越被重视"
    s8 = u"钢铁制造业是一个国家工业发展都基础"
    s9 = u"废钢收购带动了一批人致富"

    for s in [s1, s2, s3, s4, s5, s6, s7, s8, s9]:
        token_ids, segment_ids = tokenizer.encode(s, maxlen=maxlen)
        label = model.predict([[token_ids], [segment_ids]])[0].argmax()
        print("预测对标签label", label)

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('model_classify_06.weights')
            model.save('./model_classify_107.h5')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )
        # 演示效果
        just_show()
        # 我的保存
        my_keras_to_pb()


def predict_to_file(in_file, out_file):
    """输出预测结果到文件
    结果文件可以提交到 https://www.cluebenchmarks.com 评测。
    """
    fw = open(out_file, 'w')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            text = l['sentence']
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            label = model.predict([[token_ids], [segment_ids]])[0].argmax()
            l = json.dumps({'id': str(l['id']), 'label': str(label)})
            fw.write(l + '\n')
    fw.close()

if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=2,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model.weights')
    # predict_to_file('/root/CLUE-master/baselines/CLUEdataset/iflytek/test.json', 'iflytek_predict.json')
