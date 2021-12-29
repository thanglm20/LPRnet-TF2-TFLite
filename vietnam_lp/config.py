import os
from lpr.trainer import LPRVocab



input_shape = (24, 94, 3)  # (height, width, channels)
use_h_concat = False
use_oi_concat = False
max_lp_length = 20
rnn_cells_num = 128


# Path to the folder where all training and evaluation artifacts will be located
model_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
if not os.path.exists(model_dir):
  os.makedirs(model_dir)


class train:
  # Path to annotation file with training data in per line format: <path_to_image_with_license_plate label>
  file_list_path = '/media/thanglmb/Bkav/AICAM/TrainModels/OCR4ANPR/lprnet/dataset/train'

  batch_size = 32
  steps = 250000
  learning_rate = 0.001
  grad_noise_scale = 0.001
  opt_type = 'Adam'

  save_checkpoints_steps = 1000      # Number of training steps when checkpoint should be saved
  display_iter = 100

  apply_basic_aug = False
  apply_stn_aug = True
  apply_blur_aug = False

  need_to_save_weights = True
  need_to_save_log = True

  class execution:
    CUDA_VISIBLE_DEVICES = "0"  # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations


class eval:
  # Path to annotation file with validation data in per line format: <path_to_image_with_license_plate label>
  file_list_path = '/media/thanglmb/Bkav/AICAM/TrainModels/OCR4ANPR/lprnet/dataset/val'
  checkpoint = ''
  batch_size = 1

  class execution:
    CUDA_VISIBLE_DEVICES = "0"  # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations


class infer:
  # Path to text file with list of images in per line format: <path_to_image_with_license_plate>
  file_list_path = '/media/thanglmb/Bkav/AICAM/TrainModels/OCR4ANPR/lprnet/dataset/test_infer'
  checkpoint = ''
  batch_size = 1

  class execution:
    CUDA_VISIBLE_DEVICES = "0"  # Environment variable to control CUDA device used for training
    per_process_gpu_memory_fraction = 0.8  # Fix extra memory allocation issue
    allow_growth = True  # Option which attempts to allocate only as much GPU memory based on runtime allocations


vocab, r_vocab, num_classes = LPRVocab.create_vocab(train.file_list_path,
                                                    eval.file_list_path,
                                                    use_oi_concat)