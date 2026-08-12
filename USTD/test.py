# implemented by p0werHu
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.logger import Logger
from utils.wandb_utils import init_run, log_metrics, log_summary, finish
import time
from tqdm import tqdm

if __name__ == '__main__':
    opt, model_config = TestOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of samples in the dataset.
    print('The number of testing samples = %d' % dataset_size)

    # --- wandb: same per-model project as train.py, so this run sits
    # alongside its training run rather than on a shared cross-model
    # dashboard. job_type='test' distinguishes it from the training run;
    # name is suffixed so it doesn't collide with the training run's name.
    wandb_config = dict(vars(opt))
    if isinstance(model_config, dict):
        wandb_config.update(model_config)
    wandb_run = init_run(
        project=f'{opt.model}-stgnn-weather',
        group=opt.dataset_mode,
        job_type='test',
        name=f'{opt.file_time}-test',
        config=wandb_config,
    )

    model = create_model(opt, model_config)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Logger(opt)  # create a visualizer that display/save and plots
    total_iters = 0                # the total number of training iterations
    model.eval()
    val_start_time = time.time()
    try:
        for i, data in tqdm(enumerate(dataset)):  # inner loop within the test dataset
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.test()
            model.cache_results()  # store current batch results
        model.compute_visuals()  # visualization
        t_val = time.time() - val_start_time
        model.save_data()
        model.compute_metrics()
        metrics = model.get_current_metrics()
        visualizer.print_current_metrics(-1, total_iters, metrics, t_val)
        log_metrics(metrics, prefix='test')
        log_summary(metrics)
    finally:
        finish()
