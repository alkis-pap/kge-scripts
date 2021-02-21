import os, subprocess

if not os.path.isfile('link-prediction.zip'):
    subprocess.run(['wget', 'https://www.dropbox.com/s/6066tiybyro1jmh/link-prediction.zip'])
subprocess.run(['unzip', '-u', 'link-prediction.zip', '-d', 'data'])
subprocess.run(['mkdir', '-p', 'checkpoints', 'results'])


datasets = {
    'FB15K-237': [
        'data/FB15k-237/train.txt',
        'data/FB15k-237/valid.txt',
        'data/FB15k-237/test.txt'
    ],
    'WN18': [
        'data/WN18/train.txt',
        'data/WN18/valid.txt',
        'data/WN18/test.txt'
    ],
    'WN18RR': [
        'data/WN18RR/train.txt',
        'data/WN18RR/valid.txt',
        'data/WN18RR/test.txt'
    ],
    'FB15K': [
        'data/FB15k/train.txt',
        'data/FB15k/valid.txt',
        'data/FB15k/test.txt'
    ]
}