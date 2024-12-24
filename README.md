# Decima#

Simulator part of Spark(pytorch) https://hdl.handle.net/2142/121563



Example:

test scheduler  modify the provided config file config/decima_tpch.yaml and examply.py as needed
```
python  ./examples.py --sched [fair|decima|decima#]
```

If you need to train the model, you will need to adjust the configuration in the rollout_worker.py file as well as in train.py.

For training the Decima# model, modify the provided configuration file (config/scheduler_tpch.yaml) as necessary, and then provide the updated configuration file to train.py using the -f CFG_FILE option.

```
python ./train.py -f  config/scheduler_tpch.yaml
```
Use tensorboard to monitor the training process, some screenshots of the results are in `artifacts/tb/`

The scheduler's path exists  /spark_sched_sim/schedulders/neural

The existence of a path for the trainer  /trainers/


