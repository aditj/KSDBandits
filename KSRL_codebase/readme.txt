# Code Repo for our Paper : Posterior Coreset Construction with Kernelized Stein Discrepancy for Model-Based Reinforcement Learning

Please refer to requirements.txt to match the versions of the packages used.

The file structure is divided into 3 sections for each environment including Stochastic Cartpole, Stochastic Pendulum, Pusher and Reacher.

In each environment, we have 3 .py files to run various experiments for our case. 


## Stochastic Pendulum with oracle rewards: 

python run_pendulum.py --with-reward True ## this will run the code without Posterior compression and with oracle rewards
python run_pendulum_ksrl.py --with-reward True ## this will run the code for our KSRL algorithm i.e with Posterior compression

## Stochastic Pendulum without oracle rewards: 

python run_pendulum.py --with-reward False ## this will run the code without Posterior compression and without oracle rewards
python run_pendulum_ksrl.py --with-reward False ## this will run the code for our KSRL algorithm i.e with Posterior compression without oracle rewards
run_pendulum_mfree.py --algo 'sac' ## This will run model free algorithms by changing the --algo 


## Stochastic Cartpole with oracle rewards : 

python run_cartpole.py --with-reward True ## this will run the code without Posterior compression and with oracle rewards
python run_cartpole_ksrl.py --with-reward True ## this will run the code for our KSRL algorithm i.e with Posterior compression

## Stochastic Cartpole without oracle rewards : 

python run_cartpole.py --with-reward False ## this will run the code without Posterior compression and without oracle rewards
python run_cartpole_ksrl.py --with-reward False ## this will run the code for our KSRL algorithm i.e with Posterior compression without oracle rewards
run_cartpole_mfree.py --algo 'sac' ## This will run model free algorithms by changing the --algo 



## Pusher with oracle rewards : 

python run_pusher.py --with-reward True ## this will run the code without Posterior compression and with oracle rewards
python run_pusher_ksrl.py --with-reward True ## this will run the code for our KSRL algorithm i.e with Posterior compression

## Pusher without oracle rewards : 

python run_pusher.py --with-reward False ## this will run the code without Posterior compression and without oracle rewards
python run_pusher_ksrl.py --with-reward False ## this will run the code for our KSRL algorithm i.e with Posterior compression without oracle rewards
run_pusher_mfree.py --algo 'sac' ## This will run model free algorithms by changing the --algo 


## Reacher with oracle rewards : 

python run_reacher.py --with-reward True ## this will run the code without Posterior compression and with oracle rewards
python run_reacher_ksrl.py --with-reward True ## this will run the code for our KSRL algorithm i.e with Posterior compression

## Reacher without oracle rewards : 

python run_reacher.py --with-reward False ## this will run the code without Posterior compression and without oracle rewards
python run_reacher_ksrl.py --with-reward False ## this will run the code for our KSRL algorithm i.e with Posterior compression without oracle rewards
run_reacher_mfree.py --algo 'sac' ## This will run model free algorithms by changing the --algo 

Note : Please run the python files in the KSRL folder keeping all files intact. 

For saving the output files, we need to give the argument --path in the "python run_cartpole.py" files and the files will be saved in those location. For changing other parameters for ex: dictionary growth, model order, frequency just need to specify the parameters in the "NB_dx_tf_new.py" file.



