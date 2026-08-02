import numpy as np
import random, torch, os, time, json

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

class MetricsLogger(object):
    def __init__(self, fname, reinitialize=False, rename_interval=20, suffix='.jsonl'):
        self.base_name = fname
        self.fname = fname+suffix
        self.suffix = suffix
        self.reinitialize = reinitialize
        if self.reinitialize:
            for fn in os.listdir(os.path.dirname(self.base_name)):
                if fn.startswith(self.base_name) and fn.endswith(self.suffix):
                    print('{} exists, deleting...'.format(self.base_name+self.suffix))
                    os.remove(fn)
        self.cur_iter = 1
        self.rename_interval = rename_interval

    def log(self, record=None, **kwargs):
        """
        Assumption: no newlines in the input.
        """
        self.cur_iter += 1
        if record is None:
            record = {}
        record.update(kwargs)
        record['_stamp'] = time.time()
        with open(self.fname, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=True) + '\n')

    def rename(self, identity=''):
        if self.cur_iter % self.rename_interval == 0:
            new_name = self.base_name + '_' + identity + self.suffix
            if os.path.exists(self.fname): os.rename(self.fname, new_name)
            self.fname = new_name
