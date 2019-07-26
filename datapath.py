class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'siim' or dataset == 'augsiim':
            return '/home/ankoor/SIIM/siim-kaggle/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
