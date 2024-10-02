This is code for Paper "**Learned Data Transformation as A Data-centric Plugin for Enhancing Time Series Forecasting Models**" 

The template we use is https://github.com/ashleve/lightning-hydra-template

To run the code, first install with requirements.txt

```
pip install -r requirements.txt
```

Then you should specify your data paths in **.env** file 

Finally run with: 

```
python run/src/train.py task=forecasting
```


For specific configs setting, you can change the .yaml files in **configs** directories

