# Decima#

Simulator part of Spark(pytorch) https://hdl.handle.net/2142/121563



Example:

test scheduler  modify the provided config file config/decima_tpch.yaml and examply.py as needed
```
python  ./examples.py --sched [fair|decima|decima#]
```



train Decima, modify the provided config file config/decima_tpch.yaml as needed, then provide the config to train.py -f CFG_FILE.
```
python ./train.py -f  config/scheduler_tpch.yaml
```
Use tensorboard to monitor the training process, some screenshots of the results are in `artifacts/tb/`

The scheduler's path exists  /spark_sched_sim/schedulders/

The existence of a path for the trainer  /trainers/


