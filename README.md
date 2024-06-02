# metrix
Face super-resolution commonly used quantitative evaluation index index calculation warehouse, including PSNR, SSIM, LIPIS

| file          | explain                      |
| ------------- | ---------------------------- |
| calc.py       | Used to calculate PSNR, SSIM |
| calc_lipis.py | Used to calculate LIPIS      |


# Usage
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


