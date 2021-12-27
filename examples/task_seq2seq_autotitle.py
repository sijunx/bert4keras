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

from examples import modeling
from examples.my_args import arg_dic
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from keras import backend as K
from tensorflow.python.platform import gfile

# parameter ==========================
wkdir = '/Users/xusijun/Documents/NLP009/bert4keras-master001/keras_to_tensorflow-master'
pb_filename = 'model070.pb'

# 基本参数
maxlen = 256
batch_size = 16
# steps_per_epoch = 1000
steps_per_epoch = 1000
# epochs = 10000
epochs = 10

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
# txts = glob.glob('/Users/xusijun/Documents/NLP009/bert4keras-master001/MyNews/*/*.txt')

txts = glob.glob('/Users/xusijun/Documents/NLP009/bert4keras-master001/THUCNews/*/*.txt')

# 加载并精简词表，建立分词器
# token_dict, keep_tokens = load_vocab(
#     dict_path=dict_path,
#     simplified=True,
#     startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
# )
token_dict = load_vocab(
    dict_path=dict_path,
    # startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
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
    # keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    keep_tokens=None,  # 只保留keep_tokens中的字，精简原字表
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
        print("--------------------- 开始 ---------------------")
        print("prdict inputs:", inputs)
        print("prdict output_ids:", output_ids)
        print("prdict states:", states)
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        print("predict token_ids:", token_ids)
        print("predict segment_ids:", segment_ids)

        topk = 1
        proba = model.predict([token_ids, segment_ids])
        print("proba:", proba)
        log_proba = np.log(proba + 1e-6)  # 取对数，方便计算
        print("log_proba:", log_proba)

        icount =0
        maxIndex = 0
        maxValue = -9999.0
        temp = 78
        while(icount<len(proba[0][temp])):
            if(proba[0][temp][icount] > maxValue):
                maxValue = proba[0][temp][icount]
                maxIndex = icount
            icount = icount+1
        print("maxIndex:", maxIndex, " maxValue:", maxValue)
        # maxIndex: 8125  maxValue: 0.27502504
        return self.last_token(model).predict([token_ids, segment_ids])
        # print("result", scores)
        # print("states", states)
        # icount =0
        # maxIndex = 0
        # maxValue = -9999.0
        # while(icount<len(scores[0])):
        #     if(scores[0][icount] > maxValue):
        #         maxValue = scores[0][icount]
        #         maxIndex = icount
        #     icount = icount+1
        # print("maxIndex:", maxIndex, " maxValue:", maxValue)
        # print("--------------------- 结束 ---------------------")
        # return scores, states

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        # print('token_ids: ', len(token_ids), token_ids)
        # print('segment_ids: ', len(segment_ids), segment_ids)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk=topk)  # 基于beam search

        x01 = output_ids[0]
        # x02 = output_ids[1]
        # x03 = output_ids[2]
        # x04 = output_ids[3]
        # x05 = output_ids[4]

        y01 = tokenizer.decode(x01)
        print("y01:", y01)
        # y02 = tokenizer.decode(x02)
        # print("y02:", y02)
        # y03 = tokenizer.decode(x03)
        # print("y03:", y03)
        # y04 = tokenizer.decode(x04)
        # print("y04:", y04)
        # y05 = tokenizer.decode(x05)
        # print("y05:", y05)

        return tokenizer.decode(output_ids[0])


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
        # Input-Token:0
        tensor_input = sess.graph.get_tensor_by_name('Input-Token:0')
        # Input-Segment:0
        seg_input = sess.graph.get_tensor_by_name('Input-Segment:0')
        text = '夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及 时 就医 。'
        # max_c_len = maxlen - self.maxlen
        # max_c_len = maxlen - 56 + 3
        # token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        #
        # x = np.vstack((np.random.rand(1000,10),-np.random.rand(1000,10)))
        # y = np.vstack((np.ones((1000,1)),np.zeros((1000,1))))
        # x = [[2, 2352, 6702, 2234, 758, 5407, 2127, 578, 7404, 1642, 6269, 6293, 991, 670, 1399, 4393, 670, 5340, 1189, 731, 6707, 2666, 6512, 1119, 2590, 1301, 3]]
        # y = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        # token_ids, segment_ids = inputs

        x = np.array([[2, 2352, 6702, 2234, 758, 5407, 2127, 578, 7404, 1642, 6269, 6293, 991, 670, 1399, 4393, 670, 5340, 1189, 731, 6707, 2666, 6512, 1119, 2590, 1301, 3]])
        y = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        # token_ids = np.concatenate([token_ids, output_ids], 1)
        # segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        # print("predict token_ids:", token_ids)
        # print("predict segment_ids:", segment_ids)

        # print(x.shape)
        # print(y.shape)
        # predictions = sess.run(tensor_output, {tensor_input: x, seg_input: y})
        # # print('\n===== output predicted results =====\n')
        # print(predictions)
        print('xxxxxxxxxx')

def my_test001():
    text1 = '语言模型'
    # text2 = "你好"
    tokens1 = tokenizer.tokenize(text1)
    print(tokens1)
    # tokens2 = tokenizer.tokenize(text2)
    # print(tokens2)

    # indices_new, segments_new = tokenizer.encode(text1, text2, max_length=512)
    indices_new, segments_new = tokenizer.encode(text1)
    print(indices_new[:10])
    # [101, 6427, 6241, 3563, 1798, 102, 0, 0, 0, 0]
    print(segments_new[:10])
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 提取特征
    predicts_new = model.predict([np.array([indices_new]), np.array([segments_new])])[0]
    for i, token in enumerate(tokens1):
        print(token, predicts_new[i].tolist()[:5])
    # for i, token in enumerate(tokens2):
    #     print(token, predicts_new[i].tolist()[:5])

    print("xxxxx")

def my_test002():
    #加载语言模型
    #model = build_bert_model(config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True)


    token_ids, segment_ids = tokenizer.encode(u'科学技术是第一生产力')
    # mask掉“技术”
    # token_ids[3] = token_ids[4] = token_dict['[MASK]']
    token_ids[3] = token_ids[4] = token_dict['[UNK]']

    # 用mlm模型预测被mask掉的部分
    probas = model.predict([np.array([token_ids]), np.array([segment_ids])])[0]
    mask01 = tokenizer.decode(probas[3:5].argmax(axis=1))
    print(mask01) # 结果正是“技术”


    token_ids, segment_ids = tokenizer.encode(u'数学是利用符号语言研究数量、结构、变化以及空间等概念的一门学科')
    # mask掉“技术”
    #token_ids[1] = token_ids[2] = tokenizer._token_dict['[MASK]']

    # 用mlm模型预测被mask掉的部分
    probas = model.predict([np.array([token_ids]), np.array([segment_ids])])[0]
    print(tokenizer.decode(probas[1:3].argmax(axis=1))) # 结果正是“数学”

    print("xxx")

def just_show():
    # s1 = u'记者傅亚雨沈阳报道 来到沈阳，国奥队依然没有摆脱雨水的困扰。7月31日下午6点，国奥队的日常训练再度受到大雨的干扰，无奈之下队员们只慢跑了25分钟就草草收场。31日上午10点，国奥队在奥体中心外场训练的时候，天就是阴沉沉的，气象预报显示当天下午沈阳就有大雨，但幸好队伍上午的训练并没有受到任何干扰。　　下午6点，当球队抵达训练场时，大雨已经下了几个小时，而且丝毫没有停下来的意思。抱着试一试的态度，球队开始了当天下午的例行训练，25分钟过去了，天气没有任何转好的迹象，为了保护球员们，国奥队决定中止当天的训练，全队立即返回酒店。　　在雨中训练对足球队来说并不是什么稀罕事，但在奥运会即将开始之前，全队变得“娇贵”了。在沈阳最后一周的训练，国奥队首先要保证现有的球员不再出现意外的伤病情况以免影响正式比赛，因此这一阶段控制训练受伤、控制感冒等疾病的出现被队伍放在了相当重要的位置。而抵达沈阳之后，中后卫冯萧霆就一直没有训练，冯萧霆是7月27日在长春患上了感冒，因此也没有参加29日跟塞尔维亚的热身赛。队伍介绍说，冯萧霆并没有出现发烧症状，但为了安全起见，这两天还是让他静养休息，等感冒彻底好了之后再恢复训练。由于有了冯萧霆这个例子，因此国奥队对雨中训练就显得特别谨慎，主要是担心球员们受凉而引发感冒，造成非战斗减员。而女足队员马晓旭在热身赛中受伤导致无缘奥运的前科，也让在沈阳的国奥队现在格外警惕，“训练中不断嘱咐队员们要注意动作，我们可不能再出这样的事情了。”一位工作人员表示。　　从长春到沈阳，雨水一路伴随着国奥队，“也邪了，我们走到哪儿雨就下到哪儿，在长春几次训练都被大雨给搅和了，没想到来沈阳又碰到这种事情。”一位国奥球员也对雨水的“青睐”有些不解。'
    # s2 = u'新浪体育讯　主场战胜沈阳东进豪取主场六连胜的中甲劲旅延边队再传好消息。今日上午，延边州体育局与韩国认证农产品生产者协会达成赞助意向，该协会将赞助延边足球俱乐部10亿韩币(约合560万人民币)，力助延足2011赛季实现冲超。　　无偿赞助只因为延足感动此番，韩国认证农产品生产者协会为延足提供的10亿韩币赞助大单基本上都是无偿的，唯一的回报就是希望延边州体育局能够帮助该协会的产品打入延边市场做一些协调工作。说起无偿赞助延足，韩国认证农产品生产者协会中央会会长吴亨根(O Hyung-Kun)先生表示，只因延边足球让他感动。　　据吴亨根介绍，在收到延边足球俱乐部赞助提议后，他很快就做出了赞助决定。“延边足球运动员很有天赋，只要在资金上能提供有力的支持，一定会成为一流球队。”在了解了延足球员目前的训练比赛状况后，今日吴亨根还以个人名义为延边队捐了三台全自动洗衣机。　　其实，吴亨根也曾经是个足球人，他就是韩国全北现代俱乐部的创始人。1993年他创立了全北队，1994年韩国的汽车巨擘现代汽车正式入主全北队，而球队也更名成今日所用的全北现代。2006年全北现代战胜叙利亚卡马拉队夺得亚冠联赛冠军，中国球员冯潇霆目前就在这支球队效力。　　除了这10亿韩币赞助，吴亨根还表示，中甲联赛结束后，他将把延边队带到韩国进行冬训，与全北的大学生球队进行训练比赛，通过以赛代练让延足充分为下赛季实现冲超夯实基础。　　冲超早动手 经营更规范　　联赛还剩三轮，目前延边队排名第三，极有望取得征战中甲六年来的最佳战绩(此前最好排名第六)。冲超是延边队一直的梦想，延边州体育局与俱乐部方面都希望在2011赛季完成冲超大业，让延边足球重新回归国内顶级行列。要想冲超就要未雨绸缪，本赛季尚未结束，延足冲超的各项准备工作便已展开。　　本赛季延边队依然为资金所困，俱乐部经理李虎恩难辞其咎。今年7月，延边州体育局委托媒体人出身的郑宪哲先生先期运作经营延边足球俱乐部，为下赛季早作准备。年轻的郑宪哲接手后也为俱乐部经营带来了新思路，短短的两个月间，就为延足搞定了如此大单的韩国赞助意向。另外，下赛季延边队的比赛装备目前也已落实，韩国世达(STAR)体育用品公司将成为新的装备赞助商，为延足提供全套比赛训练装备，预计金额达100万人民币。　　在未来延边足球俱乐部经营思路上，延边州体育局副局长于长龙表示，要对目前俱乐部的经营进行彻底改造，以求更加适应现代足球的发展理念，在政府支持的基础上，大胆尝试市场化运作，引进韩国足球俱乐部经营运作理念，在经营、服务、宣传等方面全方位提升俱乐部的整体水平。于长龙还透露，本赛季最后一轮客场同上海东亚比赛结束后，延边足球俱乐部将在上海举行招商会，向更多企业宣传推介延边足球，实现走出去招商。而接下来，延足还将陆续前往青岛、深圳、大连等地展开招商工作。　　酝酿十年规划 打造血性之师　　据悉，延边州体育局与延边足球俱乐部近期正在酝酿推出延足未来十年的一个中长期规划，其中最首要的任务就是要在未来三年在中超站稳脚跟。如果按照这一规划的设想，至少下赛季延足要完成冲超，此后再图站稳中超。　　于长龙希望，能够在未来把延边队打造成一支文明之师、忠诚之师、血性之师、战斗之师，在继承朝鲜族球员勇猛顽强的优良传统基础上，更加彰显朝鲜族民族文化的底蕴和内涵，让延边队成为一支忠诚家乡，充满血性，真正为足球而战的足坛劲旅。　　据悉，此番敲定赞助意向只是延足为冲超迈出的第一步，如何有效转变俱乐部经营理念、如何规范运作将是摆在延边州体育局面前的一个新课题。接下来，体育局与俱乐部还将推出一系列新动作，为冲超增添筹码。　　(木木)'
    # s3 = u'日前，江苏丹阳市延陵镇的一些西瓜种植户，因为滥用膨大剂又加上暴雨，导致不少西瓜纷纷“爆炸”，并因此影响了今年的西瓜总体销量。尽管专家一再强调膨大剂本身并无危害，但是滥用的话却易引发一连串问题。花果山CEO提问：辟谣的目的是什么？消除大众对膨化剂的恐惧？继续吃膨大剂西瓜？瓜农无疑是可悲的。根本不可能自己种西瓜的消费者呢？只能吃膨大剂西瓜么？谣言粉碎机：果壳网谣言粉碎机关心的不只是给大家一个“简单的答案”。我们通过对问题的梳理，给大家提供更全面的信息，让大家能更好的做出“自己的选择”。同时，果壳网谣言粉碎机也希望向大家提供一些理性看待问题的思路。这样的思路不仅是针对一事一人的，它关涉到的是我们的生活态度与看待世界的方法。CAPA-Real-柏志提问：氯吡脲使用后，会在水果中残留吗？人体食用后对人体有些什么影响？谣言粉碎机：会有一定的残留，一般在生长初期使用，时间越长残留越少。少量的接触，对人的影响很小。具体的毒理学实验数据，果壳的文章里有详细的说明。'

    s1 = '针对最近我国赴比利时留学人员接连发生入境时被比海关拒绝或者办理身份证明时被比警方要求限期出境的事件，教育部提醒赴比利时留学人员应注意严格遵守比方相关规定。'
    # s2 = u'程序员最爱'
    # s3 = u'身体素质'

    for s in [s1]:
        print(u'生成标题:', autotitle.generate(s))
    print()

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
            model.save('./myFile70.h5')
            # model.save('/Users/xusijun/Documents/NLP009/bert4keras-master001/tensorflow-for-java-master/myFile04')
            # tf.saved_model.save(model, '/Users/xusijun/Documents/NLP009/bert4keras-master001/examples/')
            # tf.keras.models.save_model(model, '/Users/xusijun/Documents/NLP009/bert4keras-master001/examples/')

        # 演示效果
        just_show()
        # 我的保存
        my_keras_to_pb()
        # pb模型
        # save_PBmodel()

if __name__ == '__main__':

    model.load_weights('./myFile70.h5')
    just_show()

    evaluator = Evaluator()
    train_generator = data_generator(txts, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./best_model003.weights')
