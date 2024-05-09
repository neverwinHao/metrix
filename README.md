# metrix
人脸超分辨常用指标计算仓库，包含PSNR，SSIM，LIPIS
#Usage
```
python calc.py --test_dataset='Helen' --test_model='Ours'
```



```
python calc.py --test_dataset='CelebA' --test_model='SISN'
```

```
python calc_lipis.py -GT_path ./CelebA/HR -SR_path ./CelebA/SR_SISN
```

```
python calc_lipis.py -GT_path ./Helen/HR -SR_path ./Helen/SR_Ours
```


