class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/leirulin/code/biye/tracking/ltr'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.pregenerated_masks = ''
        self.lasot_dir = '/data/leirulin/Dataset/LaSOT/LaSOTBenchmark'
        self.got10k_dir = '/data/leirulin/Dataset/got10k'
        self.trackingnet_dir = ''
        self.coco_dir = '/data/leirulin/Dataset/coco'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.lasot_candidate_matching_dataset_path = ''
