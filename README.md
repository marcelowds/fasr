# Multi-Feature Aggregation in Diffusion Models for Enhanced Face Super-Resolution

## teste

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Comparison Results</title>
  
  <!-- MathJax for LaTeX rendering -->
  <script type="text/javascript" id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 20px;
      text-align: center;
    }

    figure {
      margin: 20px auto;
      width: 100%;
      max-width: 1000px;
    }

    figcaption {
      margin-top: 10px;
      font-weight: bold;
      font-size: 16px;
    }
  </style>
</head>
<body>

  <h2>Comparison Results</h2>

  <figure>
    <img src="https://raw.githubusercontent.com/marcelowds/fasr/main/fasr.png" style="width: 100%; max-width: 1000px;">
    <figcaption>
      Fig 1: Comparison of low-resolution (LR), super-resolution (SR) results obtained by various methods, 
      and ground truth (GT) images from the Quis-Campi dataset.  
      FASR outperforms baseline methods, preserving facial symmetry and natural appearance.
    </figcaption>
  </figure>

  <p>
    The low-resolution images \( \text{LR}_1, \dots, \text{LR}_N \) are used to compute a set of features 
    \( \mathbf{F}_1, \dots, \mathbf{F}_N \), respectively, which are then combined to generate \( \mathbf{F}_M \).  
    The low-resolution image \( \text{LR}_0 \) is integrated with \( \mathbf{F}_M \) in the diffusion model to produce a 
    super-resolution (SR) image.  
    The SR image is subsequently compared with a set of images from the gallery for face recognition.
  </p>

</body>
</html>


## teste

<figure>
  <img src="https://raw.githubusercontent.com/marcelowds/fasr/main/fasr_results.png" style="width: 100%; max-width: 1000px;">
  <figcaption align="center">Fig 1: Comparison of low-resolution (LR), super-resolution (SR) results obtained by various methods, and ground gruth (GT) images from the Quis-Campi
dataset. FASR outperforms baseline methods, preserving facial symmetry and natural appearance.</figcaption>
</figure>

<br><br>

This project was built using a fork of [Score-SDE](https://github.com/yang-song/score_sde) and [SDE-SR](https://github.com/marcelowds/sr-sde).

## Prepare conda environment 

```conda create -n fasr python=3.8.2```

Install requirements

```pip3 install -r requirements.txt```

Also install jax+cuda

```pip install --upgrade jax==0.2.8 jaxlib==0.1.59+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html```

Activate conda environment

```conda activate fasr```

## Tfrecords

The algorithm receives images in tfrecords format that can be generated using [Progressive Growing of GANs](https://github.com/tkarras/progressive_growing_of_gans) using:

``` python dataset_tool.py create_from_images tfrecords_path images_path --shuffle 0 ```

In the ```sample_imgs/tfrecords``` folder there is a sample of 10 images from the CelebA dataset.

## Adaface 

Download the R18 CASIA-WebFace feature extractor from [Adaface](https://github.com/mk-minchul/AdaFace?tab=readme-ov-file) [here](https://drive.google.com/file/d/1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe/view) and place it in the ```pretrained_adaface``` directory.

## Pre-trained FASR model

Download our pre-trained model [HERE](https://drive.google.com/file/d/1fPV0w2XR-svCjqkgKnOU9qpRq6GYgnpL/view?usp=drive_link) and place it in the ```exps/checkpoints-meta``` directory.

## Sample images and feature extraction

In ```sample_images```, you will find a sample of images, with gallery images in ```gallery```, low-resolution images used for feature extraction in ```LR_imgs```, probe images in high resolution in ```probe_HR```, and reference low-resolution images used for super-resolution in ```probe_LR```.

For the calculation of the mean feature, use ```features_extract.py```. Save the features in ```sample_imgs/features```.

Adjust settings and path in files ```config/default_ve_configs.py``` and ```configs/ve/sr_ve.py```.

## Generate SR images

```CUDA_VISIBLE_DEVICES=0 python3 main.py --config 'configs/ve/sr_ve.py' --mode 'sr' --workdir exps```

## Train a new model

```CUDA_VISIBLE_DEVICES=0 python3 main.py --config 'configs/ve/sr_ve.py' --mode 'train' --workdir exps```

## Citation
* DOS SANTOS, Marcelo et al. "Multi-Feature Aggregation in Diffusion Models for Enhanced Face Super-Resolution." In: *2024 37th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI)*. IEEE, 2024. p. 1-6. [[IEEE Xplore]](https://ieeexplore.ieee.org/abstract/document/10716316) [[arXiv]](https://arxiv.org/pdf/2408.15386)

```
@inproceedings{santos2024multi,
  title = {Multi-Feature Aggregation in Diffusion Models for Enhanced Face Super-Resolution},
  author = {M. {dos Santos} and R. {Laroca} and R. O. {Ribeiro} and J. {Neves} and D. {Menotti}},
  year = {2024},
  month = {Sept},
  booktitle = {Conference on Graphics, Patterns and Images (SIBGRAPI)},
  volume = {},
  number = {},
  pages = {1-6},
  doi = {10.1109/SIBGRAPI62404.2024.10716316},
  issn = {1530-1834},
}
```
