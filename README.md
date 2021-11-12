## data
- The data pth is divided into two levels, the first level is the source domain, the second level is the target domain.
    - For example, in the path './data/chip/ruijin', the source domain is chip and the target domain is ruijin.

## embedding
- ctd.50.vec download link :https://pan.baidu.com/share/init?surl=Uj97799tGjdET_vbdkW7tQ password ：vgwi
    -put ctd.50.vec to path: data/

## output
output:dset and model
  
## run
- train  

    -**source**: source domain, choose one in ruijin, chip, cars, fc    
    
    -**target**: target domain, choose one in  ruijin, chip, cars, fc, ec, nm 
    
    -**alignment**: If use domain alignment:True; Otherwise: false
    
    -**aseparation**: If use separation:True; Otherwise: false 
    
  
 ```
 python main.py  --source  ruijin  --target  chip --alignment True --separation  True  --log_file logs/log.txt --status train
 ```
  
