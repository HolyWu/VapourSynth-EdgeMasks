# EdgeMasks

Creates an edge mask using various operators.


## Parameters

```py
edgemasks.Tritical(vnode clip[, int[] planes=[0, 1, 2], float[] scale=1.0, int opt=0])
edgemasks.Cross(vnode clip[, int[] planes=[0, 1, 2], float[] scale=1.0, int opt=0])
edgemasks.Prewitt(vnode clip[, int[] planes=[0, 1, 2], float[] scale=1.0, int opt=0])
edgemasks.Sobel(vnode clip[, int[] planes=[0, 1, 2], float[] scale=1.0, int opt=0])
edgemasks.Scharr(vnode clip[, int[] planes=[0, 1, 2], float[] scale=1.0, int opt=0])
edgemasks.RScharr(vnode clip[, int[] planes=[0, 1, 2], float[] scale=1.0, int opt=0])
edgemasks.Kroon(vnode clip[, int[] planes=[0, 1, 2], float[] scale=1.0, int opt=0])
edgemasks.Robinson3(vnode clip[, int[] planes=[0, 1, 2], float[] scale=1.0, int opt=0])
edgemasks.Robinson5(vnode clip[, int[] planes=[0, 1, 2], float[] scale=1.0, int opt=0])
edgemasks.Kirsch(vnode clip[, int[] planes=[0, 1, 2], float[] scale=1.0, int opt=0])
edgemasks.ExPrewitt(vnode clip[, int[] planes=[0, 1, 2], float[] scale=1.0, int opt=0])
edgemasks.ExSobel(vnode clip[, int[] planes=[0, 1, 2], float[] scale=1.0, int opt=0])
edgemasks.FDoG(vnode clip[, int[] planes=[0, 1, 2], float[] scale=1.0, int opt=0])
edgemasks.ExKirsch(vnode clip[, int[] planes=[0, 1, 2], float[] scale=1.0, int opt=0])
```

- clip: Clip to process. Any format with either integer sample type of 8-16 bit depth or float sample type of 32 bit depth is supported. The output frames will have `_ColorRange` set to 0 (full range).

- planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.

- scale: Multiplies all pixels by `scale` before outputting. This can be used to increase or decrease the intensity of edges in the output. Can be specified for each plane individually.

- opt: Specifies which cpu optimizations to use.
  - 0 = auto detect
  - 1 = use c
  - 2 = use sse4.1
  - 3 = use avx2
  - 4 = use avx512


## Compilation

```
meson setup build
meson compile -C build
meson install -C build
```
