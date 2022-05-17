# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

from loguru import logger


def load_data(file_path):
    data = []
    labels = set()
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            if '-DOCSTART-' in line:
                continue
            terms = line.split()
            if len(terms) == 2:
                data.append([count, terms[0], terms[1]])
                labels.add(terms[1])
            else:
                count += 1
    return data, list(labels)


def generate_tsv_horizontal_bio(brand_sentences, B='B-ORG', I='I-ORG', O='O'):
    """
    Generate tsv file with horizontal bio format.
    :param brand_sentences: format: sentence '\t' brand1,brand2
    :param B:
    :param I:
    :param O:
    :return: horizontal_bio_tags: format: sentence '\t' O O O B I O
    """
    horizontal_bio_tags = []
    for line in brand_sentences:
        line = line.strip()
        terms = line.split("\t")
        sent = terms[0]
        brands = terms[1].split(',')

        tags = [O] * len(sent)
        if brands:
            # Has brands
            for brand in brands:
                brand_len = len(brand)
                if brand_len == 1:
                    continue
                brand_idx = sent.index(brand)
                if brand_idx > -1:
                    brand_start = brand_idx
                    tags[brand_start] = B
                    for i in range(brand_start + 1, brand_start + brand_len):
                        tags[i] = I
        if len(sent) != len(tags):
            logger.warning(f"sentence len not equal to tags len, sentence: {len(sent)}, tags: {len(tags)}")
            continue
        horizontal_bio_tags.append(sent + '\t' + ' '.join(tags))

    return horizontal_bio_tags


def generate_tsv_vertical_bio(brand_sentences, B='B-ORG', I='I-ORG', O='O'):
    """
    Generate tsv file with vertical bio format.
    :param brand_sentences: format: sentence '\t' brand1,brand2
    :param B:
    :param I:
    :param O:
    :return: vertical_bio_tags: output format:
        char '\t' O
        char '\t' B
        char '\t' I
    """
    vertical_bio_tags = []
    for line in brand_sentences:
        line = line.strip()
        terms = line.split("\t")
        sent = terms[0]
        brands = terms[1].split(',')

        tags = [O] * len(sent)
        if brands:
            # Has brands
            for brand in brands:
                brand_len = len(brand)
                if brand_len == 1:
                    continue
                brand_idx = sent.index(brand)
                if brand_idx > -1:
                    brand_start = brand_idx
                    tags[brand_start] = B
                    for i in range(brand_start + 1, brand_start + brand_len):
                        tags[i] = I
        if len(sent) != len(tags):
            logger.warning(f"sentence len not equal to tags len, sentence: {len(sent)}, tags: {len(tags)}")
            continue
        for i in range(len(sent)):
            vertical_bio_tags.append(sent[i] + '\t' + tags[i])
        vertical_bio_tags.append(' ')

    return vertical_bio_tags


if __name__ == '__main__':
    sents = [
        '多丽丝娃娃艾米莉7件套	艾米莉,多丽丝',
        '多丽丝娃娃 艾米莉7件套	艾米莉,多丽丝',
        '多丽丝芭比娃娃套装大礼盒BJD洋娃娃 凯蒂娃娃3分体关节60cm女孩娃娃玩具女孩玩具儿童礼物 艾米莉（12号娃娃）+梳妆7件套	艾米莉,多丽丝',
        '迪士尼（Disney）夏季立体大恐龙图案儿童短袖T恤汗衫男童半袖男宝宝卡通纯棉上衣 米白色 90	Disney,迪士尼',
    ]
    print(generate_tsv_horizontal_bio(sents))
    print(generate_tsv_vertical_bio(sents))
