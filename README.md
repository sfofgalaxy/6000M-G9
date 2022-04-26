# How to DownLoad the dataset in a linux server

- Test.zip:
  
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1JTC291uskdM0XgL6rpCXGEdq0_bgGDk9' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1JTC291uskdM0XgL6rpCXGEdq0_bgGDk9" -O test.zip && rm -rf /tmp/cookies.txt
- Train.zip:
  
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GJQ_88k7YlfXR7UWtjcAx6BlnNi1Y0m2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GJQ_88k7YlfXR7UWtjcAx6BlnNi1Y0m2" -O train.zip && rm -rf /tmp/cookies.txt
- Pretrain Anime model
  
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WNQELgHnaqMTq3TlrnDaVkyrAH8Zrjez' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WNQELgHnaqMTq3TlrnDaVkyrAH8Zrjez" -O anime.pkl && rm -rf /tmp/cookies.txt

# How to uncompress the huge size .zip file

1. install 7za
   1. wget
   2. uncompress document
   3. make
   4. edit the line of "DECT_DIR" line in install.sh
   5. ./install.sh
   6. or just use sudo apt-get if you are in superdoers
2. 7za x [filename] -r -o./

# Stardize Dataset

python dataset_tool.py --source=../../data --dest=../../data/anime-128x128.zip --resolution=128x128

python avg_spectra.py calc --source=../../data/anime-128x128.zip --dest=tmp/training-data.npz --mean=134.437 --std=75.2062

python avg_spectra.py calc --source=./runs/00006-stylegan3-t-anime-128x128-gpus4-batch32-gamma8.2/network-snapshot-000400.pkl --dest=./runs/stylegan3-t.npz --mean=134.437 --std=75.2062 --num=70000

# Training

python train.py --outdir=./runs --cfg=stylegan3-t --data=../../data/anime-128x128.zip --gpus=4 --batch=32 --gamma=8.2 --mirror=1 --resume=./runs/00002-stylegan3-t-anime-128x128-gpus4-batch32-gamma8.2/network-snapshot-00xxxx.pkl

# How to install GCC without root user

[install GCC (style GAN required)](https://blog.csdn.net/qq_29750461/article/details/104885031)

1. tar -xvf gcc-7.5.0.tar.gz && cd gcc-7.5.0
2. ./contrib/download_prerequisites
3. mkdir objdir && cd objdir
4. ../configure --disable-checking --enable-languages=c,c++ --disable-multilib --prefix=/home/zpeng/install/gcc-7.5.0 --enable-threads=posix
5. make -j64 && make install (多线程编译)
6. export PATH=/home/zpeng/install/gcc-7.5.0/bin:/home/zpeng/install/gcc-7.5.0/lib64:$PATH && export LD_LIBRARY_PATH=/home/zpeng/install/gcc-7.5.0/lib/:$LD_LIBRARY_PATH && source ~/.bashrc

# generate image and video with stylegan3

python gen_images.py --outdir=out --trunc=1 --seeds=0713 --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 --network=./runs/00002-stylegan3-t-anime-128x128-gpus4-batch32-gamma8.2/network-snapshot-000200.pkl

- anime:

python gen_images.py --outdir=out --trunc=1 --seeds=3516 --network=/home/zpeng/.cache/dnnlib/downloads/anime.pkl

# generate images using diffusion model

Setting up environment

`cd ./improved-diffusion`
`pip install -e .`
`pip install mpi4py`

Training

```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3" DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --microbatch 16 --class_cond False"
OPENAI_LOGDIR="path/to/checkpoint"
```

`python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS`
Three models will be created after the training, these three models should all be preserved if continue training is needed.

Sampling

EMA model is recommended for sampling, which usually presentes a better results.
Pretrained model for this project can be found in https://drive.google.com/file/d/11Fsu62DqmQQu-Jh3tLHsN2jxCXLDiPA4/view?usp=sharing

```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --microbatch 16 --class_cond False"
OPENAI_LOGDIR="path/to/sample"
```

`python scripts/image_sample.py --model_path path/to/checkpoint/ema_0.9999_103000.pt --batch_size 4 --num_samples 40 --timestep_respacing 250 $MODEL_FLAGS $DIFFUSION_FLAGS`
After the sampling, a samples_num_samplesx64x64x3.npz file will be created,where `arr_0` in the file is the collection of sample images.



You can find more detail information in ./improved-diffusion/README.md
