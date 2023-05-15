#!/usr/bin/python3
import argparse
import os
import sys
from typing import List

import numpy as np
import time
import webbrowser
from PIL import Image
from jinja2 import FileSystemLoader, Environment
import clickhouse_connect
import clip
import torch
from pyparsing import *
from dataclasses import dataclass

ppc = pyparsing_common


def _search(client, table, column, features, limit=10):
    st = time.time()
    order = "ASC"
    columns = ['url', 'caption', f'L2Distance({column},{features}) AS score', column]
    query = f'SELECT {",".join(columns)} FROM {table} ORDER BY score {order} LIMIT {limit}'
    print(query)
    result = client.query(query)
    et = time.time()
    rows = [{
        'url': row[0],
        'caption': row[1],
        'score': round(row[2], 3),
        'embedding': row[3]
    }
        for row in result.result_rows]
    return rows, {'read_rows': result.summary['read_rows'], 'query_time': round(et - st, 3)}


def search_with_text(client, model, table, text, limit=10):
    st = time.time()
    inputs = clip.tokenize(text)
    with torch.no_grad():
        text_features = model.encode_text(inputs)[0].tolist()
        et = time.time()
        rows, stats, = _search(client, table, 'image_embedding', text_features, limit=limit)
        stats['generation_time'] = round(et - st, 3)
        return rows, stats


def search_with_images(client, model, table, image_url, limit=10):
    st = time.time()
    image = preprocess(Image.open(image_url)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)[0].tolist()
        et = time.time()
        rows, stats = _search(client, table, 'text_embedding', image_features, limit=limit)
        stats['generation_time'] = round(et - st, 3)
        return rows, stats


@dataclass
class Expression:
    text: str


@dataclass
class Feature:
    vector: List[float]


def concept_to_expression(concept):
    if type(concept) == str:
        with torch.no_grad():
            return Feature(model.encode_text(clip.tokenize(concept))[0].tolist())
    elif type(concept) == int or type(concept) == Expression or type(concept) == Feature:
        return concept
    else:
        return text_concepts_to_vector(model, concept)


def concept_to_str(concept):
    if type(concept) == Feature:
        return f'[{",".join(map(str, concept.vector))}]'
    elif type(concept) == Expression:
        return concept.text
    return str(concept)


def text_concepts_to_vector(model, concepts):
    if len(concepts) != 3:
        raise 'unbalanced expressions - must be "<concept> <operator> <concept>"'
    left_concept = concept_to_expression(concepts[0])
    operator = concepts[1]
    if operator not in ['+', '-', '/', '*']:
        raise f'{operator} is not a valid operator'
    right_concept = concept_to_expression(concepts[2])
    if type(left_concept) == int and type(right_concept):
        raise 'unbalanced expressions - must be "<concept> <operator> <concept>"'
    if type(left_concept) == int:
        return Expression(f'arrayMap(x -> {left_concept}{operator}x, {concept_to_str(right_concept)})')
    elif type(right_concept) == int:
        return Expression(f'arrayMap(x -> x{operator}{right_concept}, {concept_to_str(left_concept)})')
    else:
        return Expression(f'arrayMap((x,y) -> x{operator}y, {concept_to_str(left_concept)}, '
                          f'{concept_to_str(right_concept)})')


def text_inflix_expression_to_vector(client, model, table, concepts, limit=10):
    st = time.time()
    features = text_concepts_to_vector(model, concepts[0])
    et = time.time()
    # this will be an image embedding, use this to find similar images based on caption
    rows, stats = _search(client, table, 'image_embedding', features.text, limit=limit)
    stats['generation_time'] = round(et - st, 3)
    return rows, stats


def link(uri, label=None):
    if label is None:
        label = uri
    parameters = ''
    escape_mask = '\033]8;{};{}\033\\{}\033]8;;\033\\'
    return escape_mask.format(parameters, uri, label)


ParserElement.enablePackrat()
sys.setrecursionlimit(3000)

word = Word(alphas)
phrase = QuotedString("'", escChar='\\')
integer = ppc.integer

operand = word | phrase | integer
plusop = oneOf("+ -")
signop = oneOf("+ -")
multop = oneOf("* /")

expr = infixNotation(
    operand,
    [
        (multop, 2, opAssoc.LEFT),
        (plusop, 2, opAssoc.LEFT),
    ],
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='search',
        description='Search for matching images in the Laion dataset by either text or image')

    client = clickhouse_connect.get_client(host=os.environ.get('CLICKHOUSE_HOST', 'localhost'),
                                           username=os.environ.get('CLICKHOUSE_USERNAME', 'default'),
                                           password=os.environ.get('CLICKHOUSE_PASSWORD', ''),
                                           port=os.environ.get('CLICKHOUSE_PORT', 8443),
                                           secure=True if os.environ.get('CLICKHOUSE_SSL', 'True') == 'True' else False)
    sub_parsers = parser.add_subparsers(dest='command')

    search_parser = sub_parsers.add_parser('search', help='search using text or images')
    search_parser_params = search_parser.add_mutually_exclusive_group(required=True)
    search_parser_params.add_argument('--text', required=False)
    search_parser_params.add_argument('--image', required=False)

    concept_parser = sub_parsers.add_parser('concept_math', help='blend two concepts from images')
    concept_parser.add_argument('--text', required=True)

    parser.add_argument('--table', default='laion_100m')
    parser.add_argument('--limit', default=10)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model, preprocess = clip.load("ViT-L/14")
    model.to(device)
    images = []
    stats = {}
    text = None
    image = None
    if args.command == 'search':
        if args.text:
            text = args.text
            images, stats = search_with_text(client, model, args.table, text, limit=args.limit)
        else:
            image = args.image
            images, stats = search_with_images(client, model, args.table, image, limit=args.limit)
    elif args.command == 'concept_math':
        text = args.text
        concepts = expr.parseString(args.text)
        images, stats = text_inflix_expression_to_vector(client, model, args.table, concepts, limit=args.limit)

    environment = Environment(loader=FileSystemLoader("templates/"))
    template = environment.get_template("results.html")
    filename = f"results_{int(time.time())}.html"
    context = {
        "images": images,
        "table": args.table,
        "text": text,
        "source_image": image,
        "gen_time": stats['generation_time'],
        "query_time": stats['query_time'],
    }
    with open(filename, mode="w", encoding="utf-8") as message:
        message.write(template.render(context))
    file_link = f"file://{os.path.join(os.getcwd(), filename)}"
    print(link(file_link))
    webbrowser.open_new(file_link)
