#!/bin/bash
for num_models in 1 2 4
do
  for input_len in 96 512
  do
      for output_len in 96 192 336 720
      do
          python src/train.py data=ETTh1 data.input_len=$input_len data.output_len=$output_len task=forecasting model.net.num_models=$num_models
          python src/train.py data=ETTh2 data.input_len=$input_len data.output_len=$output_len task=forecasting model.net.num_models=$num_models
          python src/train.py data=ETTm1 data.input_len=$input_len data.output_len=$output_len task=forecasting model.net.num_models=$num_models
          python src/train.py data=ETTm2 data.input_len=$input_len data.output_len=$output_len task=forecasting model.net.num_models=$num_models
          python src/train.py data=electricity data.input_len=$input_len data.output_len=$output_len task=forecasting model.net.num_models=$num_models
          python src/train.py data=traffic data.input_len=$input_len data.output_len=$output_len task=forecasting model.net.num_models=$num_models
          python src/train.py data=weather data.input_len=$input_len data.output_len=$output_len task=forecasting model.net.num_models=$num_models
      done
  done
done
