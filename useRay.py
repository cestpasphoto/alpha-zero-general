#!/usr/bin/env python3

from ray import train, tune, shutdown, init
import argparse
# import sys
from os import environ
# from time import sleep
# from random import random
from ray_config import myfunc, gen_params

nb_threads = 6
temp_dir = '/home/best/ray_temp/'

def gen_tune_config(args, param_space, param_bound):
	if args.scheduler == 'PBT':
		from ray.tune.schedulers import PopulationBasedTraining       
		scheduler = PopulationBasedTraining(
			time_attr="training_iteration",
			perturbation_interval=1,
			metric="adjusted_score",
			mode="max",
			hyperparam_mutations=param_space,
			# quantile_fraction=0.333,
			# resample_probability=0.,
			synch=False,
		)
		tune_config = tune.TuneConfig(num_samples=nb_threads, scheduler=scheduler)

	elif args.scheduler == 'PB2':
		from ray.tune.schedulers.pb2 import PB2        
		scheduler = PB2(
			time_attr="training_iteration",
			perturbation_interval=1,
			metric="adjusted_score",
			mode="max",
			hyperparam_bounds=param_bound,
			quantile_fraction=0.333,
			synch=True,
		)
		tune_config = tune.TuneConfig(num_samples=nb_threads, scheduler=scheduler)

	elif args.scheduler == 'ASHA':
		from ray.tune.schedulers import ASHAScheduler
		tune_config=tune.TuneConfig(
			metric='adjusted_score',
			mode='max',
			scheduler=ASHAScheduler(grace_period=1, max_t=4), # regarde la perf à partir de 2 itérations, fin train à 8
			num_samples=50  # population de 20 items (au début)
		)

	else:
		raise Exception(f'scheduler {args.scheduler} not known')

	return tune_config

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Apply Ray Tune to Alphazero-based training')
	parser.add_argument('--trial'           ,        action='store', required=True    , help='Trial/experiment name')
	parser.add_argument('--scheduler'       ,        action='store', default='PBT'    , help='PBT or PB2 or ASHA')
	parser.add_argument('--synch'           ,        action='store_true'              , help='Enable synch option in scheduler')
	parser.add_argument('--init-dir'        ,        action='store', required=True    , help='Folder where is initial NN')
	parser.add_argument('--comp-dir'        ,        action='store', default=None     , help='NN to compare each experiment with')
	# handle relative dirs
	# auto trial name ?
	#   -m -e -p -d -V -P -ppit
	#   params to enable/disable
	#   params space
	args = parser.parse_args()
	if args.comp_dir is None:
		args.comp_dir = args.init_dir

	# Disable memory checks
	environ["RAY_memory_monitor_refresh_ms"] = "0"
	run_config=train.RunConfig(name=args.trial, verbose=0, stop={'training_iteration': 35})
	param_init, param_space, param_bound = gen_params(args)
	tune_config = gen_tune_config(args, param_space, param_bound)    
	tuner = tune.Tuner(lambda config: myfunc(config, args, temp_dir), run_config=run_config, tune_config=tune_config, param_space=param_init)
	# tuner = tune.Tuner.restore(path='/home/best/ray_results/'+config_name, trainable=myfunc)

	shutdown()
	init(include_dashboard=False, _temp_dir=temp_dir)
	results = tuner.fit()
