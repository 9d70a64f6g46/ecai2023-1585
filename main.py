import os
import json
import pickle
from typing import Iterable
from operator import concat
from functools import reduce
from stanfordcorenlp import StanfordCoreNLP
from config import *


# configure java environment
os.environ['PATH'] = f'{os.environ["PATH"]}:{JAVA_PATH}/bin:{JAVA_PATH}/jre/bin'


# noinspection PyMethodMayBeStatic
class AOSExtractor:
    def __init__(self, target: str):
        self.acos_file = f'{BASE_DIR}/data/acos/{target}_quad_test.tsv'
        self.shap_file = f'{BASE_DIR}/data/aos/{target}_quad_test_shaps.pkl'
        self.sentiment_file = f'{BASE_DIR}/data/aos/{target}_quad_test_sentiment.pkl'
        self.pos_map = {
            "UH": "n", "NNS": "n", "NNPS": "n", "NN": "n", "NNP": "n", "VBN": "v", "VB": "v", "VBD": "v", "VBZ": "v",
            "VBP": "v", "VBG": "v", "JJR": "a", "JJS": "a", "JJ": "a", "RBS": "r", "RB": "r", "RP": "r", "WRB": "r", "RBR": "r",
        }
        # load sentiwordnet
        sentiwordnet = dict()
        with open(f'{BASE_DIR}/data/dict/SentiWordNet_3.0.0.txt', 'r', encoding='utf-8') as f:
            for line in map(lambda x: x.split('\t'), f.read().splitlines()[26:-1]):
                for word_number in map(lambda x: x.split('#'), line[4].split(' ')):
                    sentiwordnet.setdefault(word_number[0], {}).setdefault(line[0], []).append([int(word_number[1]), float(line[2]), float(line[3])])
        self.sentiwordnet = {word: {pos: sum(map(lambda x: abs(x[1] + x[2]) / x[0], score_list)) for pos, score_list in info.items()} for word, info in sentiwordnet.items()}
        # laod dataset
        self.data_list = self._load_data()

    def get_senti(self, word: str, pos: str):
        """
        get sentiwordnet score

        :param word: the word
        :param pos: part of speech
        :return: return -1 if not exists
        """
        return self.sentiwordnet.get(word, {}).get(self.pos_map.get(pos, ''), -1)

    def extract_top_k_opinion(self, extract_implicit: bool, threshold: float, k: int):
        """
        extract the top k highest shap value as opinion
        """
        opinion_list = []
        for data in self.data_list:
            words_info = sorted(filter(lambda x: self.pos_map.get(x[1], None) is not None, data['words_info']), key=lambda x: x[3], reverse=True)
            if len(words_info) == 0:
                opinion_list.append([[-1, -1]])
                continue
            temp = list(filter(lambda x: x[3] > 0, words_info))
            if len(temp) == 0:
                temp.append(words_info[0])
            opinion = map(lambda x: [-1, -1] if extract_implicit and self.get_senti(x[0], x[1]) < threshold else [x[4][0], x[4][-1] + 1], temp)
            opinion_list.append(list(opinion))
        return opinion_list

    def extract_sigma_opinion(self, extract_implicit: bool, threshold: float, sigma: float):
        """
        use z-score standard the shap value then filtered abnormally high closeness
        """
        from sklearn import preprocessing

        opinion_list = []
        for data in self.data_list:
            words_info = sorted(filter(lambda x: self.pos_map.get(x[1], None) is not None, data['words_info']), key=lambda x: x[3], reverse=True)
            if len(words_info) == 0:
                opinion_list.append([[-1, -1]])
                continue
            result = list(map(lambda x: x[1], filter(lambda x: x[0] > sigma, zip(preprocessing.scale(list(map(lambda x: x[3], words_info))), words_info))))
            if len(result) == 0:
                opinion_list.append([[-1, -1]])
                continue
            opinion = map(lambda x: [-1, -1] if extract_implicit and self.get_senti(x[0], x[1]) < threshold else [x[4][0], x[4][-1] + 1], result)
            opinion_list.append(list(opinion))
        return opinion_list

    def extract_aspect(self, opinion_list: list):
        with open(f'{BASE_DIR}/data/aos/temp/opinion.txt', 'w', encoding='utf-8') as f:
            for data, opinion in zip(self.data_list, opinion_list):
                f.write(data['sentence'])
                f.write('\t')
                f.write('; '.join(map(lambda x: ','.join(map(str, x)), opinion)))
                f.write('\n')
        os.system(f'java -jar {BASE_DIR}/data/aos/SentiAspectExtractor-0.0.1-SNAPSHOT-jar-with-dependencies.jar  -inputfile {BASE_DIR}/data/aos/temp/opinion.txt -outputfile {BASE_DIR}/data/aos/temp/aspect.txt -dict {BASE_DIR}/data/aos/dictionary/')
        with open(f'{BASE_DIR}/data/aos/temp/aspect.txt', 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_aos(self, aspect_list: list, opinion_list: list):
        return list(map(lambda x: {'sentence': x[2]['sentence'], 'aos_list': list(map(lambda y: {'index': y['opinion_index'], 'aspect': y['aspect'], 'opinion': x[1][y['opinion_index']], 'sentiment': x[2]['sentiment']}, x[0]))}, zip(aspect_list, opinion_list, self.data_list)))

    def _is_true(self, range1: list, range2: list, which: str):
        range1 = [range1[0], range1[1] - 1] if range1[0] != -1 else range1
        range2 = [range2[0], range2[1] - 1] if range2[0] != -1 else range2
        if which == 'strict':
            return range1[0] == range2[0] and range1[1] == range2[1]
        elif which == 'loose':
            return range1[0] <= range2[0] <= range1[1] or range2[0] <= range1[0] <= range1[1] <= range2[1] or range1[0] <= range2[1] <= range1[1]
        raise RuntimeError('illegal argument')

    def _quota(self, aos_list: list, aos_map, data_map, is_true):
        TP = FP = GTP = GFN = 0
        for pred_aos, real_aos in zip(map(lambda x: set(map(aos_map, x['aos_list'])), aos_list),
                                      map(lambda x: set(map(data_map, x['acso_list'])), self.data_list)):
            pred_result = list(map(lambda x: len(list(filter(lambda y: is_true(x, y), real_aos))) > 0, pred_aos))
            real_result = list(map(lambda x: len(list(filter(lambda y: is_true(x, y), pred_aos))) > 0, real_aos))
            TP += sum(pred_result)
            FP += len(pred_result) - sum(pred_result)
            GTP += sum(real_result)
            GFN += len(real_result) - sum(real_result)
        P, R = TP / (TP + FP), GTP / (GTP + GFN)
        F = (2 * P * R) / (P + R)
        return P, R, F

    def quota_aos(self, aos_list: list, which: str):
        return self._quota(aos_list, lambda x: (tuple(x['aspect']), tuple(x['opinion']), x['sentiment']), lambda x: (tuple(x[0]), tuple(x[3]), x[2] - 1), lambda x, y: self._is_true(x[0], y[0], which) and self._is_true(x[1], y[1], which) and x[2] == y[2])

    def quota_as(self, aos_list: list, which: str):
        return self._quota(aos_list, lambda x: (tuple(x['aspect']), x['sentiment']), lambda x: (tuple(x[0]), x[2] - 1), lambda x, y: self._is_true(x[0], y[0], which) and x[1] == y[1])

    def quota_a(self, aos_list: list, which: str):
        return self._quota(aos_list, lambda x: (tuple(x['aspect']),), lambda x: (tuple(x[0]),), lambda x, y: self._is_true(x[0], y[0], which))

    def quota_o(self, aos_list: list, which: str):
        return self._quota(aos_list, lambda x: (tuple(x['opinion']),), lambda x: (tuple(x[3]),), lambda x, y: self._is_true(x[0], y[0], which))

    def _process_sentence(self, sentence: str) -> str:
        index, words, result = 0, sentence.split(' '), []
        while index < len(words):
            if index < len(words) - 3 and words[index+1] == "'" and words[index+2] in ('t', 'd', 'm', 's', 're', 've', 'll'):
                result.append(''.join(words[index:index+3]))
                index += 3
                continue
            result.append(words[index])
            index += 1
        return ' '.join(result)

    def _index_token(self, sentence: str, token_list: Iterable):
        index, result = 0, []
        for token in token_list:
            token_index = sentence.find(token[0], index)
            if token_index < 0:
                return []
            index = token_index + len(token[0])
            result.append((token[0], token[1], (token_index, index)))
        return result

    def _merge(self, iter_pos: list, iter_lemma: list, iter_shap: list, iter_index: list):
        if 0 in (len(iter_pos), len(iter_lemma), len(iter_shap), len(iter_index)):
            return []
        return list(map(lambda x: (x[0][0], x[0][1], x[1], sum(map(lambda y: y[1], filter(lambda y: self._is_true(x[0][2], y[2], 'loose'), iter_shap))), list(map(lambda y: y[1], filter(lambda y: self._is_true(x[0][2], y[2], 'loose'), iter_index)))), zip(iter_pos, iter_lemma)))

    def _load_data(self):
        data_list = []
        with open(self.acos_file, 'r', encoding='utf-8') as f:
            acso_list = list(map(lambda y: [y[0], *map(lambda a: [list(map(int, a[0].split(','))), a[1], int(a[2]), list(map(int, a[3].split(',')))], map(lambda z: z.split(' '), y[1:]))], map(lambda x: x.split('\t'), f.read().splitlines())))
        with open(self.shap_file, 'rb') as f:
            shap_list = pickle.load(f)
        with open(self.sentiment_file, 'rb') as f:
            sentiment_list = pickle.load(f)
        with StanfordCoreNLP(STANFORD_CORE_NLP_PATH) as nlp:
            for acso, shaps, sentiment in zip(acso_list, shap_list, sentiment_list):
                sentence = self._process_sentence(acso[0])
                iter_lemma = list(map(lambda x: x['lemma'], chain(*map(lambda x: x['tokens'], json.loads(nlp.annotate(sentence))['sentences']))))
                iter_pos, iter_shap, iter_index = nlp.pos_tag(sentence), map(lambda x: (x[0].strip(), x[1]), shaps), map(lambda x: (x[1], x[0]), enumerate(acso[0].split(' ')))
                iter_pos, iter_shap, iter_index = self._index_token(sentence, iter_pos), self._index_token(sentence, iter_shap), self._index_token(sentence, iter_index)
                data_list.append({'sentence': acso[0], 'processed_sentence': sentence, 'sentiment': sentiment, 'dependency': nlp.dependency_parse(sentence), 'acso_list': acso[1:], 'words_info': self._merge(iter_pos, iter_lemma, iter_shap, iter_index)})
        return data_list


def main():
    """
    second step work
    """
    extractor = AOSExtractor('laptop')
    aspect_list = extractor.extract_aspect(extractor.extract_top_k_opinion(False, 0, 1))
    opinion_list = extractor.extract_top_k_opinion(True, 0.2, 1)
    aos_list = extractor.extract_aos(aspect_list, opinion_list)
    result = extractor.quota_aos(aos_list, 'loose')
    print(f'P:{result[0]},R:{result[1]},F:{result[2]}')


if __name__ == "__main__":
    main()
