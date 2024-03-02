from ray import tune, train
import tempfile
from math import log10
from subprocess import run
from os.path import isfile

# learn_prefix = './main.py -m 400 -e 200 -p 2 -i 2 -s 2 -n 2 -F -d 1.0 -T 10 -V 31 -P 4 --useray'
# pit_prefix = './pit.py -m 400 -n 100 --useray'
learn_prefix = '/home/best/dev/splendor/alpha-zero-general/main.py -m 600 -e 100 -p 2 -i 2 -s 2 -n 2 -F -d 1.0 -T 10 -V 31 -P 4 --useray'
pit_prefix = '/home/best/dev/splendor/alpha-zero-general/pit.py -m 400 -n 50 --useray'

def core_func(config, prev_dir, next_dir, comp_dir, temp_dir):
	if not isfile(f'{prev_dir}/temp.pt'):
		print(f'{prev_dir}/temp.pt NOT FOUND')
	pt = 'best.pt' if isfile(f'{prev_dir}/best.pt') else 'temp.pt'

	bsz, lr, tHigh = 2**max(config["log_bsz"], 3), round(max(config["learn_rate"], 1e-5), 5), round(config["temp"], 3)
	tLow = round(1/tHigh, 3)
	learn_args = f'-c {config["cpuct"]:.3} -f {config["fpu"]:.3} -D {config["dropout"]:.3} -b {bsz} -l {lr} -q {config["q_weight"]:.3} -t {tHigh} {tLow} -u {config["universes"]}'
	learn_cmdline = learn_prefix.split(' ')
	learn_cmdline += ['-L', prev_dir+'/'+pt, '-C', next_dir]
	learn_cmdline += learn_args.split(' ')
	print(' '.join(learn_cmdline))
	run(learn_cmdline)

	pt = 'best.pt' if isfile(f'{next_dir}/best.pt') else 'temp.pt'
	pit_cmdline = pit_prefix.split(' ') + [comp_dir+'/best.pt', next_dir+'/'+pt]
	print(' '.join(pit_cmdline))
	run(pit_cmdline)
	with open(f'{next_dir}/score.txt', 'r') as f:
		score = f.read()
		score = float(score)
	return score

def myfunc(config, args, temp_dir):
	chkpt = train.get_checkpoint()
	chkpt_dir0 = chkpt.to_directory() if chkpt else args.init_dir
	prev_dir = chkpt_dir0
	temp_dir = tempfile.TemporaryDirectory(prefix=args.trial[:4]+'_', dir=temp_dir)

	for step in range(35):
		next_dir = f'{temp_dir.name}/{step:02}'
		score = core_func(config, prev_dir, next_dir, args.comp_dir, temp_dir)
		penalty = min(log10(config["learn_rate"])+5, 0)*5 + min(config["log_bsz"]-3, 0)*5 + min(config["temp"]-1.1, 0)*25
		metrics = {'adjusted_score': score + penalty, 'true_score': score}
		metrics.update(config)
		train.report(metrics=metrics, checkpoint=train.Checkpoint.from_directory(next_dir))
		prev_dir = next_dir

def gen_params(args):
	param_space={
		'cpuct'     : tune.uniform(0.3  , 1.0 ),
		#'fpu'       : 0.2,
		'dropout'   : tune.uniform(0.1  , 0.5 ),
		# 'log_bsz'   : tune.randint(4    , 10  ),
		# 'learn_rate': tune.uniform(3e-5 , 3e-4),
		'q_weight'  : tune.uniform(0.4 , 0.80 ),
		# 'temp'      : tune.uniform(1.1  , 1.4 ),
		'universes' : tune.randint(1    , 6   ),
	}

	param_bounds = {
	  'cpuct'     : [1.00 , 1.75 ],
	  'fpu'       : [0.02 , 0.04 ],
	  'dropout'   : [0.1  , 0.25 ],
	  'log_bsz'   : [5    , 8    ],
	  'learn_rate': [-5.  , -3.  ],
	  'q_weight'  : [0.30 , 0.60 ],
	  'temp'      : [1.1  , 1.4 ],
	  'universes' : 1,
	}

	param_init = {
		'cpuct'     : tune.uniform(0.3 , 1.0 ),
		'fpu'       : 0.2,
		'dropout'   : 0.25,
		'log_bsz'   : 5,   # tune.randint(5    , 8  ),
		'learn_rate': 1e-5,
		'q_weight'  : tune.uniform(0.4 , 0.8 ),
		'temp'      : 1.25,
		'universes' : tune.randint(1    , 4  ),
	}

	return param_init, param_space, param_bounds
