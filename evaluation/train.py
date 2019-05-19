from library import *

# load pre-processed data bundle - javadoc for train set, android for test set
data = LanguagePairDataset('bundle', use_cuda=torch.cuda.is_available(),
                           src_maxlen=300, dst_maxlen=15)

# load extraction results from our approach as the test set
test_label = np.array([int(line.strip())
                       for line in open('results.txt')])
test_train = np.random.rand(len(data.test)) > 0.3
test_holdout = np.zeros(len(data.test), np.bool)
for i in range(0, 90000, 300):
    test_holdout[i] = True

# set batch size and number of dimensions
bs = 100
nd = 300

# parameter grid for two options: model architecture and training dataset
grid = dict(
    model = [
        lambda: Matcher(
            Encoder(len(data.src_vocab), nd),
            Encoder(len(data.dst_vocab), nd),
            n_enc = nd,
            n_hid = nd,
        ),
        lambda: Seq2Seq(
            Encoder(len(data.src_vocab), nd),
            Encoder(len(data.dst_vocab), nd),
        )
    ],
    trainset = [
        ('javadoc', data.train, data.test),
        ('label1_test',
             data.test[(test_label == 1) & (~test_holdout)],
             data.test[test_holdout],
        ),
        ('random_test',
             data.test[test_train & (~test_holdout)],
             data.test[test_holdout],
        ),
    ],
)

# training options: starting and ending epoch
# start from a positive number to resume previous training
n_epoch_start = 0
n_epoch_end = 10

if not os.path.isdir('checkpoints'):
    os.mkdir('checkpoints')

for i, (model_gen, (trainset_name, trainset, testset)) in enumerate(itertools.product(*grid.values())):
    model = model_gen()
    if data.use_cuda:
        model.cuda()
    model_name =  f'Model_{model.__class__.__name__}-trainset_{trainset_name}'
    print('=' * 80)
    print(i + 1, model_name)
    print('=' * 80)
    for name, dataset in [('train', trainset), ('test', testset)]:
        x, y, m = data.batch(np.random.choice(dataset, bs))
        print('-' * 20, 'sample', name, 'data', '-' * 20)
        print('>> src')
        pprint(data.src_vocab, x[0])
        print('>> dst')
        pprint(data.dst_vocab, y[0])
        print()
    print('train/test size', len(trainset), len(testset))
    for epoch in range(n_epoch_start, n_epoch_end):
        print()
        print('Epoch', epoch + 1, flush=True)
        fname = f'checkpoints/{model_name}-epoch_{epoch + 1}'
        if os.path.isfile(fname):
            model = torch.load(fname, map_location='cuda' if data.use_cuda else 'cpu')
            continue
        # train
        optim = torch.optim.Adam(model.parameters())
        model.train()
        if isinstance(model, Matcher):
            data.negative_sample = 0.5
        else:
            data.negative_sample = 0
        data.one_epoch(model, shuffled(trainset), bs, optim=optim)
        torch.save(model, fname)
        # eval
        model.eval()
        if isinstance(model, Matcher):
            idx = data.test[test_holdout]
            x, y, m = data.batch(idx)
            y_ = model.forward(x, y)
            s = roc_auc_score(test_label[test_holdout], y_.cpu().detach().numpy().reshape(-1))
            print('roc_auc_score', s)
        else:
            idx = data.test[test_holdout][test_label[test_holdout]==1]
            x, y, m = data.batch(idx)
            y_ = model.translate(x, data.dst_maxlen)
            s = np.mean([bleu(y[i], y_[i]) for i in range(bs)])
            print('bleu score', s)
    print()
