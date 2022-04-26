# DCGAN
## Create Environment and install modules
```.bash
cd ./DCGAN
conda create -n dc python=3.8.8
conda activate dc
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install matplotlib
```
## Train DCGAN
```
python train.py
```

# StyleGAN3

## Computing Environment Configuration
### How to DownLoad dataset from Google Drive
```.bash
# Test.zip
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download& \
  confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \ 
  'https://docs.google.com/uc?export=download&id=1JTC291uskdM0XgL6rpCXGEdq0_bgGDk9' -O- | \ 
  sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1JTC291uskdM0XgL6rpCXGEdq0_bgGDk9" -O test.zip \
  && rm -rf /tmp/cookies.txt
# Train.zip:
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download& \
  confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
  'https://docs.google.com/uc?export=download&id=1GJQ_88k7YlfXR7UWtjcAx6BlnNi1Y0m2' -O- | \
  sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GJQ_88k7YlfXR7UWtjcAx6BlnNi1Y0m2" -O train.zip \ 
  && rm -rf /tmp/cookies.txt
```

### How to uncompress the huge size .zip file

```.bash
# install 7za
   1. wget
   2. uncompress document
   3. make
   4. edit the line of "DECT_DIR" line in install.sh
   5. ./install.sh
   6. or just use sudo apt-get if you are in superdoers
# uncompress
7za x [filename] -r -o./
```

### How to install GCC without root user

[install GCC (style GAN required)](https://blog.csdn.net/qq_29750461/article/details/104885031)

```.bash
tar -xvf gcc-7.5.0.tar.gz && cd gcc-7.5.0
./contrib/download_prerequisites
mkdir objdir && cd objdir
../configure --disable-checking --enable-languages=c,c++ --disable-multilib --prefix=/home/zpeng/install/gcc-7.5.0 --enable-threads=posix
make -j64 && make install (multitread compile)
export PATH=/home/zpeng/install/gcc-7.5.0/bin:/home/zpeng/install/gcc-7.5.0/lib64:$PATH && export LD_LIBRARY_PATH=/home/zpeng/install/gcc-7.5.0/lib/:$LD_LIBRARY_PATH && source ~/.bashrc
```

## Quick Start
### Create Environment and install modules
```.bash
cd ./StyleGAN3
conda env create -f environment.yml
conda activate stylegan3
```
### Pre-process dataset
```.bash
# 128x128 resolution.
python dataset_tool.py --source=/tmp/images --dest=../../data/images-128x128.zip
```
### Train StyleGAN3
```.bash
python train.py --outdir=./runs --cfg=stylegan3-t --data=../../data/images-128x128.zip \
    --gpus=8 --batch=32 --gamma=8.2 --mirror=1 [--resume=*/*.pkl]
```
### Generate Images and Videos
```.bash
# Generate an image using pre-trained model
python gen_images.py --outdir=out --trunc=1 --seeds=6000 --network=*/*.pkl
# Render a 4x2 grid of interpolations for seeds 0 through 31.
python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 --network=*/*.pkl
```
### Spectral analysis
```.bash
# Calculate dataset mean and std, needed in subsequent steps.
python avg_spectra.py stats --source=../../data/images-128x128.zip 

# Calculate average spectrum for the training data.
python avg_spectra.py calc --source=../../data/images-128x128.zip --dest=tmp/training-data.npz --mean=134.437 --std=75.2062

# Calculate average spectrum for a pre-trained generator.
python avg_spectra.py calc --source=*/*.pkl --dest=tmp/stylegan3-t.npz --mean=134.437 --std=75.2062 --num=70000

# Display results.
python avg_spectra.py heatmap tmp/training-data.npz
python avg_spectra.py heatmap tmp/stylegan3-t.npz
python avg_spectra.py slices tmp/training-data.npz tmp/stylegan3-t.npz
```
## More details
### data_tool.py
  The input dataset format is guessed from the --source argument:
```
  --source *_lmdb/                    Load LSUN dataset
  --source cifar-10-python.tar.gz     Load CIFAR-10 dataset
  --source train-images-idx3-ubyte.gz Load MNIST dataset
  --source path/                      Recursively load all images from path/
  --source dataset.zip                Recursively load all images from dataset.zip
```
  Specifying the output format and path:
```
  --dest /path/to/dir                 Save output files under /path/to/dir
  --dest /path/to/dataset.zip         Save output files into /path/to/dataset.zip
```
  The output dataset format can be either an image folder or an uncompressed
  zip archive. Zip archives makes it easier to move datasets around file
  servers and clusters, and may offer better training performance on network
  file systems.

  Images within the dataset archive will be stored as uncompressed PNG.
  Uncompresed PNGs can be efficiently decoded in the training loop.

  Class labels are stored in a file called 'dataset.json' that is stored at
  the dataset root folder.  This file has the following structure:
```JSON
  {
      "labels": [
          ["00000/img00000000.png",6],
          ["00000/img00000001.png",9],
          ... repeated for every image in the datase
          ["00049/img00049999.png",1]
      ]
  }
```
  If the 'dataset.json' file cannot be found, the dataset is interpreted as
  not containing class labels.

  Image scale/crop and resolution requirements:

  Output images must be square-shaped and they must all have the same power-
  of-two dimensions.

  To scale arbitrary input image size to a specific width and height, use
  the --resolution option.  Output resolution will be either the original
  input resolution (if resolution was not specified) or the one specified
  with --resolution option.

  Use the --transform=center-crop or --transform=center-crop-wide options to
  apply a center crop transform on the input image.  These options should be
  used with the --resolution option.  For example:

  python dataset_tool.py --source LSUN/raw/cat_lmdb --dest /tmp/lsun_cat \
      --transform=center-crop-wide --resolution=512x384

Options:
```
  --source PATH                   Directory or archive name for input dataset [required]
  --dest PATH                     Output directory or archive name for output dataset  [required]
  --max-images INTEGER            Output only up to `max-images` images
  --transform                     [center-crop|center-crop-wide], Input crop/resize mode
  --resolution WxH                Output resolution (e.g., '512x512')
  --help                          Show this message and exit.
```

### train.py
 Examples:
```.bash
  # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
  python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \
      --gpus=8 --batch=32 --gamma=8.2 --mirror=1

  # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
  python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \
      --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \
      --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

  # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
  python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \
      --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
```
Options:
```
  --outdir DIR                    Where to save the results  [required]
  --cfg [stylegan3-t|stylegan3-r|stylegan2]
                                  Base configuration  [required]
  --data [ZIP|DIR]                Training data  [required]
  --gpus INT                      Number of GPUs to use  [required]
  --batch INT                     Total batch size  [required]
  --gamma FLOAT                   R1 regularization weight  [required]
  --cond BOOL                     Train conditional model  [default: False]
  --mirror BOOL                   Enable dataset x-flips  [default: False]
  --aug [noaug|ada|fixed]         Augmentation mode  [default: ada]
  --resume [PATH|URL]             Resume from given network pickle
  --freezed INT                   Freeze first layers of D  [default: 0]
  --p FLOAT                       Probability for --aug=fixed  [default: 0.2]
  --target FLOAT                  Target value for --aug=ada  [default: 0.6]
  --batch-gpu INT                 Limit batch size per GPU
  --cbase INT                     Capacity multiplier  [default: 32768]
  --cmax INT                      Max. feature maps  [default: 512]
  --glr FLOAT                     G learning rate  [default: varies]
  --dlr FLOAT                     D learning rate  [default: 0.002]
  --map-depth INT                 Mapping network depth  [default: varies]
  --mbstd-group INT               Minibatch std group size  [default: 4]
  --desc STR                      String to include in result dir name
  --metrics [NAME|A,B,C|none]     Quality metrics  [default: fid50k_full]
  --kimg KIMG                     Total training duration  [default: 25000]
  --tick KIMG                     How often to print progress  [default: 4]
  --snap TICKS                    How often to save snapshots  [default: 50]
  --seed INT                      Random seed  [default: 0]
  --fp32 BOOL                     Disable mixed-precision  [default: False]
  --nobench BOOL                  Disable cuDNN benchmarking  [default: False]
  --workers INT                   DataLoader worker processes  [default: 3]
  -n, --dry-run                   Print training options and exit
  --help                          Show this message and exit.
```

# Diffusion Model

Setting up environment

## Create Environment and install modules
```
cd ./improved-diffusion
pip install -e .
pip install mpi4py
```

## Training

```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3" DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --microbatch 16 --class_cond False"
OPENAI_LOGDIR="path/to/checkpoint"
```

```
python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```
Three models will be created after the training, these three models should all be preserved if continue training is needed.

## Sampling

EMA model is recommended for sampling, which usually presentes a better results.
Pretrained model for this project can be found in https://drive.google.com/file/d/11Fsu62DqmQQu-Jh3tLHsN2jxCXLDiPA4/view?usp=sharing

```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --microbatch 16 --class_cond False"
OPENAI_LOGDIR="path/to/sample"
```

```
python scripts/image_sample.py --model_path path/to/checkpoint/ema_0.9999_103000.pt \
--batch_size 4 --num_samples 40 --timestep_respacing 250 $MODEL_FLAGS $DIFFUSION_FLAGS
```
After the sampling, a samples_num_samplesx64x64x3.npz file will be created,where `arr_0` in the file is the collection of sample images.



You can find more detail information in ./improved-diffusion/README.md
