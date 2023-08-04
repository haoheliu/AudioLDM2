# AudioLDM 2

A proper choice of seed is important for model performance.

## Installation

You will need to have python >= 3.7 and CUDA support to run AudioLDM2 

```shell
    pip install audioldm2=0.0.2
```

## Sound effect and Music Generation

- Generate based on a text prompt

```shell
audioldm2 -t "Musical constellations twinkling in the night sky, forming a cosmic melody."
```

- Generate based on a list of text

```shell
audioldm2 -tl batch.lst
```

